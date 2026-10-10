"""Pilot: what does Whisper's 30-s padding do?  Encoder registers vs decoder-side end-of-speech evidence.

Runs a hand-written Whisper encoder (same weights/maths as HF, but any length and optional prefix KV)
plus greedy decoding, on short LibriSpeech test-clean utterances, under these input conditions:

  pad30        standard: log-mel padded to 30 s (1500 encoder frames)            [baseline]
  nopad        encoder sees only the utterance (T = dur*50 frames)
  padK         utterance + K s of zero padding (K in --pad-secs)
  nopad+tail   encoder on the utterance only; decoder additionally sees the encoder outputs of the
               padding positions T..1499 taken from a *reference* padded run of a different utterance
               (training-free "silence caching" without block masks)
  nopad+reg    encoder on the utterance only, with R register tokens prefixed as per-layer KV that were
               cached from the reference padded run (RegCache-style transplant); decoder sees speech frames only
  nopad+reg+tail   both
  pad30-mask   standard padded input, but encoder self-attention to the R top-norm (register) frames is
               masked from the emergence layer on (necessity test)

Also records, for pad30, per-layer token-norm outlier statistics and where the top-norm tokens sit.
Indicative only (small n); see plan.md section "Pilot" for how results are used.
"""
import argparse, glob, json, math, os, random, re, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000


def load_librispeech(root, n, min_s, max_s, seed):
    trans = {}
    for f in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        d = os.path.dirname(f)
        for line in open(f, encoding="utf-8"):
            uid, txt = line.strip().split(" ", 1)
            trans[os.path.join(d, uid + ".flac")] = txt
    items = []
    for p in sorted(trans):
        info = sf.info(p)
        dur = info.frames / info.samplerate
        if min_s <= dur <= max_s:
            items.append((p, trans[p], dur))
    random.Random(seed).shuffle(items)
    return items[:n]


class HandEncoder:
    """Whisper encoder forward written out explicitly (matches HF WhisperEncoder in eval mode)."""

    def __init__(self, enc):
        self.e = enc
        self.H = enc.layers[0].self_attn.num_heads
        self.scale = enc.layers[0].self_attn.scaling

    def embed(self, feats):
        x = F.gelu(self.e.conv1(feats))
        x = F.gelu(self.e.conv2(x)).permute(0, 2, 1)
        T = x.shape[1]
        return x + self.e.embed_positions.weight[:T].unsqueeze(0)

    def run(self, x, prefix_kv=None, mask_keys=None, mask_from=0, record=False, keep_kv=False):
        """prefix_kv: list[(k,v)] per layer with shape (1,H,R,dh) or None.
        mask_keys: LongTensor of key positions to mask (encoder self-attention) for layers >= mask_from."""
        B, T, D = x.shape
        H = self.H; dh = D // H
        rec, kvs = [], []
        for i, L in enumerate(self.e.layers):
            if record:
                rec.append(x[0].float())
            h = L.self_attn_layer_norm(x)
            a = L.self_attn
            q = (a.q_proj(h) * self.scale).view(B, T, H, dh).transpose(1, 2)
            k = a.k_proj(h).view(B, T, H, dh).transpose(1, 2)
            v = a.v_proj(h).view(B, T, H, dh).transpose(1, 2)
            if keep_kv:
                kvs.append((k.detach(), v.detach()))
            if prefix_kv is not None and prefix_kv[i] is not None:
                pk, pv = prefix_kv[i]
                k = torch.cat([pk, k], 2); v = torch.cat([pv, v], 2)
            logits = q @ k.transpose(-1, -2)
            if mask_keys is not None and i >= mask_from:
                off = 0 if prefix_kv is None or prefix_kv[i] is None else prefix_kv[i][0].shape[2]
                logits[..., mask_keys + off] = float("-inf")
            o = (logits.float().softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, T, D)
            x = x + a.out_proj(o)
            h = L.final_layer_norm(x)
            x = x + L.fc2(F.gelu(L.fc1(h)))
        if record:
            rec.append(x[0].float())  # pre-final-LN output of last block
        return self.e.layer_norm(x), rec, kvs


@torch.no_grad()
def greedy(model, tok, enc_out, max_len=160):
    dev = enc_out.device
    prompt = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    ids = torch.tensor([prompt], device=dev)
    past, out = None, []
    for _ in range(max_len):
        o = model.model.decoder(input_ids=ids if past is None else ids[:, -1:], encoder_hidden_states=enc_out,
                                past_key_values=past, use_cache=True)
        past = o.past_key_values
        lg = model.proj_out(o.last_hidden_state[:, -1]).float()
        lg[:, eot + 1:] = float("-inf")
        nxt = int(lg.argmax(-1))
        if nxt == eot:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
    return tok.decode(out, skip_special_tokens=True)


def logmel(fe, audio, pad_to_samples):
    """Whisper log-mel for `audio` zero-padded to pad_to_samples (multiple of 320 so T is integral)."""
    pad_to_samples = int(math.ceil(pad_to_samples / 320) * 320)
    x = np.zeros(pad_to_samples, np.float32); x[: len(audio)] = audio[:pad_to_samples]
    f = fe(x, sampling_rate=SR, return_tensors="pt", padding="do_not_pad").input_features
    return f[..., : pad_to_samples // 160]


def outlier_stats(rec, T_audio):
    """Per layer: max/median token norm, Sun ratio (max|x|/median|x|) with |x| floor, share of top-norm in padding."""
    out = []
    for l, x in enumerate(rec):
        n = x.norm(dim=-1); med = n.median()
        a = x.abs(); psi = float(a.max() / a.median())
        top = int(n.argmax())
        out.append(dict(layer=l, norm_ratio=round(float(n.max() / med), 2), psi=round(psi, 1), max_abs=round(float(a.max()), 1),
                        n_out10=int((n > 10 * med).sum()), top=top, top_in_pad=bool(top >= T_audio)))
    return out


def is_halluc(hyp, ref):
    h, r = hyp.split(), ref.split()
    rep = bool(re.search(r"\b(\w+\s+\w+\s+\w+)\b(?:\s+\1\b){1,}", hyp))
    return (len(h) > len(r) + 3) or rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--data", default="data/LibriSpeech/test-clean")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-s", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=8.0)
    ap.add_argument("--pad-secs", default="1,2,5,10")
    ap.add_argument("--regs", type=int, default=4, help="number of register frames to transplant / mask")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="experiments/pilot_padding/results")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if args.fp16 else torch.float32
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dt).to(dev).eval()
    fe, tok = proc.feature_extractor, proc.tokenizer
    norm = tok.normalize if hasattr(tok, "normalize") else (lambda s: s.lower())
    enc = HandEncoder(model.model.encoder)
    items = load_librispeech(args.data, args.n + 1, args.min_s, args.max_s, seed=0)
    ref_item, items = items[0], items[1:]

    # ---- reference padded run: cache register KV (top-norm frames at the emergence layer) and padding-tail outputs
    ra, _ = sf.read(ref_item[0], dtype="float32")
    rf = logmel(fe, ra, 30 * SR).to(dev, dt)
    r_out, r_rec, r_kv = enc.run(enc.embed(rf), record=True, keep_kv=True)
    T_ref = int(len(ra) / SR * 50)
    stats_ref = outlier_stats(r_rec, T_ref)
    ratios = [s["norm_ratio"] for s in stats_ref]
    emerge = next((i for i, r in enumerate(ratios) if r > 10), int(np.argmax(ratios)))
    nrm = r_rec[min(emerge + 1, len(r_rec) - 1)].norm(dim=-1)
    reg_pos = torch.topk(nrm, args.regs).indices.sort().values
    prefix = [(k[:, :, reg_pos], v[:, :, reg_pos]) for (k, v) in r_kv]
    print(f"[ref] emergence layer={emerge}  register frames={reg_pos.tolist()}  (ref audio frames={T_ref})")

    pad_secs = [float(s) for s in args.pad_secs.split(",") if s]
    conds = ["pad30", "nopad"] + [f"pad{int(k)}" for k in pad_secs] + ["nopad+tail", "nopad+reg", "nopad+reg+tail", "pad30-mask"]
    hyps = {c: [] for c in conds}; refs = []; frames = {c: [] for c in conds}; census = []
    t0 = time.time()
    for idx, (p, txt, dur) in enumerate(items):
        a, _ = sf.read(p, dtype="float32")
        refs.append(norm(txt))
        f30 = logmel(fe, a, 30 * SR).to(dev, dt)
        out30, rec30, _ = enc.run(enc.embed(f30), record=(idx < 20))
        T = int(math.ceil(len(a) / 320))  # encoder frames covering the audio
        if idx < 20:
            census.append(outlier_stats(rec30, T))
        hyps["pad30"].append(norm(greedy(model, tok, out30))); frames["pad30"].append(1500)
        fnp = logmel(fe, a, len(a)).to(dev, dt)
        xnp = enc.embed(fnp)
        outnp, _, _ = enc.run(xnp)
        hyps["nopad"].append(norm(greedy(model, tok, outnp))); frames["nopad"].append(xnp.shape[1])
        for k in pad_secs:
            fk = logmel(fe, a, min(len(a) + int(k * SR), 30 * SR)).to(dev, dt)
            xk = enc.embed(fk); ok, _, _ = enc.run(xk)
            hyps[f"pad{int(k)}"].append(norm(greedy(model, tok, ok))); frames[f"pad{int(k)}"].append(xk.shape[1])
        Tn = xnp.shape[1]
        tail = r_out[:, Tn:1500]
        hyps["nopad+tail"].append(norm(greedy(model, tok, torch.cat([outnp, tail], 1)))); frames["nopad+tail"].append(Tn)
        outr, _, _ = enc.run(xnp, prefix_kv=prefix)
        hyps["nopad+reg"].append(norm(greedy(model, tok, outr))); frames["nopad+reg"].append(Tn + args.regs)
        hyps["nopad+reg+tail"].append(norm(greedy(model, tok, torch.cat([outr, tail], 1)))); frames["nopad+reg+tail"].append(Tn + args.regs)
        # necessity: mask this utterance's own top-norm frames (found in its pad30 run) from the emergence layer on
        _, recm, _ = enc.run(enc.embed(f30), record=True)
        own = torch.topk(recm[min(emerge + 1, len(recm) - 1)].norm(dim=-1), args.regs).indices
        outm, _, _ = enc.run(enc.embed(f30), mask_keys=own, mask_from=emerge)
        hyps["pad30-mask"].append(norm(greedy(model, tok, outm))); frames["pad30-mask"].append(1500)
        if idx % 10 == 0:
            print(f"{idx}/{len(items)}  {time.time()-t0:.0f}s  ref='{refs[-1][:60]}'  nopad='{hyps['nopad'][-1][:60]}'")

    res = dict(model=args.model, n=len(refs), regs=args.regs, emergence_layer=emerge, ref_register_frames=reg_pos.tolist(),
               ref_utt=os.path.basename(ref_item[0]), stats_ref=stats_ref, census_first20=census, results={})
    print(f"\n== {args.model}  n={len(refs)}  durations {args.min_s}-{args.max_s}s  regs={args.regs}")
    for c in conds:
        wer = jiwer.wer(refs, hyps[c]) * 100
        hal = np.mean([is_halluc(h, r) for h, r in zip(hyps[c], refs)]) * 100
        fr = float(np.mean(frames[c]))
        res["results"][c] = dict(wer=round(wer, 2), halluc_pct=round(hal, 1), mean_encoder_frames=round(fr, 1))
        print(f"{c:16s} WER={wer:6.2f}%  halluc={hal:5.1f}%  enc_frames={fr:7.1f}")
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, args.model.split("/")[-1] + f"_n{len(refs)}_r{args.regs}.json")
    res["examples"] = [dict(ref=r, **{c: hyps[c][i] for c in conds}) for i, r in enumerate(refs[:15])]
    json.dump(res, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
