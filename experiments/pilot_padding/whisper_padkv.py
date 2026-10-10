"""Pilot v2: is Whisper's padding an attention-sink reservoir?  Cached-padding KV for padding-free encoding.

Builds on whisper_regs.py (same hand-written encoder). A content-free *silence-only* 30-s input is encoded once;
its per-layer keys/values ("padding KV") and outputs ("padding tail") are cached. For each utterance the encoder
then runs ONLY on the speech frames (T = dur*50); speech queries additionally attend to cached padding keys/values
at the positions the padding would have occupied ([T, 1500)). Padding frames never attend to speech (one-way cache).

Conditions
  pad30                 standard (1500 frames)
  nopad                 speech frames only
  pad5                  speech + 5 s real zero padding
  tail                  encoder on speech only; decoder sees speech outputs + cached padding tail
  kv                    speech attends to cached padding KV (all of [T,1500)); decoder sees speech outputs only
  kv+tail               both (the candidate training-free method)
  kvTOP{k}+tail         only the k cached padding frames that receive the most attention in the silence-only run
  kvRND{k}+tail         k random cached padding frames (control for kvTOP)
  kv+tailTOP{k}         decoder sees speech outputs + only the k most-attended padding-tail frames
  pad30-dropTOP{k}      standard encoding, but the k highest-norm frames are removed from the decoder's view
Reports WER, hallucination rate, and analytic encoder FLOPs relative to pad30.
"""
import argparse, json, math, os, time
import numpy as np, soundfile as sf, torch, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_regs import HandEncoder, greedy, logmel, load_librispeech, is_halluc, SR


def enc_flops(cfg, Tq, Tk):
    d, f, L = cfg.d_model, cfg.encoder_ffn_dim, cfg.encoder_layers
    per_layer = Tq * (4 * d * d + 2 * d * f) * 2 + 2 * Tq * Tk * d * 2
    return L * per_layer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--data", default="data/LibriSpeech/test-clean")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-s", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=8.0)
    ap.add_argument("--ks", default="4,16,64,256")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="experiments/pilot_padding/results")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if a.fp16 else torch.float32
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
    fe, tok = proc.feature_extractor, proc.tokenizer
    cfg = model.config
    enc = HandEncoder(model.model.encoder)
    ks = [int(k) for k in a.ks.split(",") if k]

    # ---- cache: silence-only 30 s input
    sil = logmel(fe, np.zeros(30 * SR, np.float32), 30 * SR).to(dev, dt)
    xs = enc.embed(sil)
    s_out, s_rec, s_kv = enc.run(xs, record=True, keep_kv=True)
    # attention received per frame in the silence-only run (mean over layers, heads, queries)
    recv = torch.zeros(1500, device=dev)
    xr = xs
    for i, L in enumerate(enc.e.layers):
        h = L.self_attn_layer_norm(xr); at = L.self_attn
        H = enc.H; dh = cfg.d_model // H
        q = (at.q_proj(h) * enc.scale).view(1, -1, H, dh).transpose(1, 2)
        k = at.k_proj(h).view(1, -1, H, dh).transpose(1, 2)
        A = (q @ k.transpose(-1, -2)).float().softmax(-1)
        recv += A.mean((0, 1, 2))
        v = at.v_proj(h).view(1, -1, H, dh).transpose(1, 2)
        o = (A.to(v.dtype) @ v).transpose(1, 2).reshape(1, -1, cfg.d_model)
        xr = xr + at.out_proj(o); h2 = L.final_layer_norm(xr); xr = xr + L.fc2(torch.nn.functional.gelu(L.fc1(h2)))
    sil_norms = [float(r.norm(dim=-1).max() / r.norm(dim=-1).median()) for r in s_rec]
    print("[silence-only] per-layer max/median norm:", [round(x, 1) for x in sil_norms])
    print("[silence-only] top-8 attended frames:", torch.topk(recv, 8).indices.tolist())

    items = load_librispeech(a.data, a.n, a.min_s, a.max_s, seed=0)
    conds = ["pad30", "nopad", "pad5", "tail", "kv", "kv+tail"] + [f"kvTOP{k}+tail" for k in ks] + [f"kvRND{k}+tail" for k in ks] \
        + [f"kv+tailTOP{k}" for k in (4, 16)] + [f"pad30-dropTOP{k}" for k in (4, 16)] + [f"pad30-encmaskTOP{k}" for k in (4, 16)]
    # emergence layer = first block whose output max/median token norm exceeds 10 in the silence-only run
    emerge = next((i for i, r in enumerate(sil_norms[1:]) if r > 10), int(np.argmax(sil_norms[1:])))
    print("[silence-only] emergence block:", emerge)
    hyps = {c: [] for c in conds}; flops = {c: [] for c in conds}; refs = []
    rng = np.random.default_rng(0)
    t0 = time.time()
    for idx, (p, txt, dur) in enumerate(items):
        au, _ = sf.read(p, dtype="float32")
        refs.append(tok.normalize(txt))
        f30 = logmel(fe, au, 30 * SR).to(dev, dt)
        o30, rec30, _ = enc.run(enc.embed(f30), record=True)
        hyps["pad30"].append(tok.normalize(greedy(model, tok, o30))); flops["pad30"].append(enc_flops(cfg, 1500, 1500))
        xn = enc.embed(logmel(fe, au, len(au)).to(dev, dt)); T = xn.shape[1]
        on, _, _ = enc.run(xn)
        hyps["nopad"].append(tok.normalize(greedy(model, tok, on))); flops["nopad"].append(enc_flops(cfg, T, T))
        x5 = enc.embed(logmel(fe, au, min(len(au) + 5 * SR, 30 * SR)).to(dev, dt)); o5, _, _ = enc.run(x5)
        hyps["pad5"].append(tok.normalize(greedy(model, tok, o5))); flops["pad5"].append(enc_flops(cfg, x5.shape[1], x5.shape[1]))
        tail = s_out[:, T:1500]
        hyps["tail"].append(tok.normalize(greedy(model, tok, torch.cat([on, tail], 1)))); flops["tail"].append(enc_flops(cfg, T, T))
        pos = torch.arange(T, 1500, device=dev)
        prefix_all = [(k[:, :, pos], v[:, :, pos]) for (k, v) in s_kv]
        okv, _, _ = enc.run(xn, prefix_kv=prefix_all)
        hyps["kv"].append(tok.normalize(greedy(model, tok, okv))); flops["kv"].append(enc_flops(cfg, T, 1500))
        hyps["kv+tail"].append(tok.normalize(greedy(model, tok, torch.cat([okv, tail], 1)))); flops["kv+tail"].append(enc_flops(cfg, T, 1500))
        rv = recv[T:1500]
        for k in ks:
            k_eff = min(k, 1500 - T)
            top = pos[torch.topk(rv, k_eff).indices]
            rnd = pos[torch.as_tensor(rng.choice(len(pos), k_eff, replace=False), device=dev)]
            for name, sel in ((f"kvTOP{k}+tail", top), (f"kvRND{k}+tail", rnd)):
                pk = [(kk[:, :, sel], vv[:, :, sel]) for (kk, vv) in s_kv]
                ok, _, _ = enc.run(xn, prefix_kv=pk)
                hyps[name].append(tok.normalize(greedy(model, tok, torch.cat([ok, tail], 1)))); flops[name].append(enc_flops(cfg, T, T + k_eff))
        for k in (4, 16):
            topt = torch.topk(rv, min(k, 1500 - T)).indices
            hyps[f"kv+tailTOP{k}"].append(tok.normalize(greedy(model, tok, torch.cat([okv, tail[:, topt]], 1)))); flops[f"kv+tailTOP{k}"].append(enc_flops(cfg, T, 1500))
            nrm = rec30[-1].norm(dim=-1)
            drop = torch.topk(nrm, k).indices
            keep = torch.ones(1500, dtype=torch.bool, device=dev); keep[drop] = False
            hyps[f"pad30-dropTOP{k}"].append(tok.normalize(greedy(model, tok, o30[:, keep]))); flops[f"pad30-dropTOP{k}"].append(enc_flops(cfg, 1500, 1500))
            # encoder-side necessity: mask self-attention to the top-norm frames (found at the emergence block) from there on
            em = rec30[min(emerge + 1, len(rec30) - 1)].norm(dim=-1)
            om, _, _ = enc.run(enc.embed(f30), mask_keys=torch.topk(em, k).indices, mask_from=emerge)
            hyps[f"pad30-encmaskTOP{k}"].append(tok.normalize(greedy(model, tok, om))); flops[f"pad30-encmaskTOP{k}"].append(enc_flops(cfg, 1500, 1500))
        if idx % 20 == 0:
            print(f"{idx}/{len(items)} {time.time()-t0:.0f}s ref='{refs[-1][:50]}' kv+tail='{hyps['kv+tail'][-1][:50]}'")

    base = np.mean(flops["pad30"])
    res = dict(model=a.model, n=len(refs), silence_norm_ratio_by_layer=sil_norms, results={})
    print(f"\n== {a.model} n={len(refs)} ({a.min_s}-{a.max_s}s)")
    for c in conds:
        w = jiwer.wer(refs, hyps[c]) * 100; hal = np.mean([is_halluc(h, r) for h, r in zip(hyps[c], refs)]) * 100
        fr = np.mean(flops[c]) / base
        res["results"][c] = dict(wer=round(w, 2), halluc_pct=round(hal, 1), rel_encoder_flops=round(fr, 3))
        print(f"{c:18s} WER={w:7.2f}%  halluc={hal:5.1f}%  encFLOPs={fr:5.3f}x")
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, a.model.split("/")[-1] + f"_padkv_n{len(refs)}.json")
    res["examples"] = [dict(ref=r, **{c: hyps[c][i] for c in conds}) for i, r in enumerate(refs[:15])]
    json.dump(res, open(fn, "w"), indent=1); print("saved", fn)


if __name__ == "__main__":
    main()
