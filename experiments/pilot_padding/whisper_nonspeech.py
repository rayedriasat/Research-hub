"""Pilot v3: does Whisper hallucinate less on non-speech audio when it can 'see' silence?

ESC-50 clips (5 s, no speech). Whisper should output nothing. Conditions (greedy, <|en|><|transcribe|><|notimestamps|>):
  pad30          clip + 25 s zero padding (standard)
  nopad          clip only
  nopad+tail     clip only; decoder also sees the cached silence-only encoder tail (positions T..1499)
  tile30         clip tiled to fill 30 s (no silence anywhere)
  tile25+sil5    clip tiled to 25 s + 5 s zero padding
Metric: hallucination rate = share of clips with >= 1 output word after Whisper normalisation; mean words per clip.
"""
import argparse, csv, json, os, random
import numpy as np, soundfile as sf, torch, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_regs import HandEncoder, greedy, logmel, SR


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--esc", default="../../data/ESC-50-master")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dev = "cuda"; dt = torch.float16 if a.fp16 else torch.float32
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
    fe, tok = proc.feature_extractor, proc.tokenizer
    enc = HandEncoder(model.model.encoder)
    s_out, _, _ = enc.run(enc.embed(logmel(fe, np.zeros(30 * SR, np.float32), 30 * SR).to(dev, dt)))
    rows = list(csv.DictReader(open(os.path.join(a.esc, "meta", "esc50.csv"))))
    random.Random(0).shuffle(rows); rows = rows[: a.n]
    conds = ["pad30", "nopad", "nopad+tail", "tile30", "tile25+sil5"]
    outs = {c: [] for c in conds}; cats = []
    for r in rows:
        x, sr = sf.read(os.path.join(a.esc, "audio", r["filename"]), dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        cats.append(r["category"])
        o30, _, _ = enc.run(enc.embed(logmel(fe, x, 30 * SR).to(dev, dt)))
        outs["pad30"].append(tok.normalize(greedy(model, tok, o30)))
        xn = enc.embed(logmel(fe, x, len(x)).to(dev, dt)); T = xn.shape[1]
        on, _, _ = enc.run(xn)
        outs["nopad"].append(tok.normalize(greedy(model, tok, on)))
        outs["nopad+tail"].append(tok.normalize(greedy(model, tok, torch.cat([on, s_out[:, T:1500]], 1))))
        t30 = np.tile(x, int(np.ceil(30 * SR / len(x))))[: 30 * SR]
        ot, _, _ = enc.run(enc.embed(logmel(fe, t30, 30 * SR).to(dev, dt)))
        outs["tile30"].append(tok.normalize(greedy(model, tok, ot)))
        t25 = t30[: 25 * SR]
        o25, _, _ = enc.run(enc.embed(logmel(fe, t25, 30 * SR).to(dev, dt)))
        outs["tile25+sil5"].append(tok.normalize(greedy(model, tok, o25)))
    res = dict(model=a.model, n=len(rows), results={})
    print(f"== {a.model}  ESC-50 n={len(rows)}")
    for c in conds:
        nw = [len(o.split()) for o in outs[c]]
        rate = float(np.mean([w > 0 for w in nw])) * 100
        res["results"][c] = dict(halluc_rate=round(rate, 1), mean_words=round(float(np.mean(nw)), 2))
        print(f"{c:12s} halluc={rate:5.1f}%  mean_words={np.mean(nw):6.2f}")
    res["examples"] = [dict(category=cats[i], **{c: outs[c][i] for c in conds}) for i in range(20)]
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, a.model.split('/')[-1] + f"_esc50_n{len(rows)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
