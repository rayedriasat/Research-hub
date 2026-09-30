# Gap Re-Verification & A*-Readiness Assessment — "Do Audio Transformers Need Registers?"

**Date:** 2026-09-30 · **Supersedes the gap claims in** `ResearchPlan.md` §2/§9 (last verified 2026-07-16)
**Method:** (1) Semantic Scholar citation-graph crawl of 16 anchor papers (≈4,200 citing papers) filtered for audio/speech; (2) 17 Semantic Scholar keyword searches; (3) 22 arXiv API field-restricted queries incl. everything submitted Jul–Sep 2026; (4) targeted web searches (Zenodo, OpenReview, GitHub, IEEE); (5) **a CPU pilot experiment on 7 pretrained audio encoders** to test whether the phenomenon exists at all.
Raw outputs: `verification/` (search logs) and `pilot/` (code, per-model JSON, full log).

---

## TL;DR verdict

**Yes — still worth pursuing, and the pilot says the phenomenon is real and large in the models that matter most. But the plan must be re-framed before it can reach an A\* venue.**

1. **The gap is still open for speech encoders** (HuBERT, WavLM, Whisper encoder, wav2vec2/XLS-R, BEATs, EAT, Audio-MAE). Zero of ~4,200 papers citing the 16 anchors characterise artifacts/registers in a pretrained audio encoder.
2. **One partial pre-emption appeared (June 2026):** a Zenodo course-project preprint, *"Silence in the Noise: Probing Attention Artifacts and Register Tokens in Audio Spectrogram Transformers"* (S. Imdad, JHU), reports that **AST has no high-norm artifacts** (max/median norm 1.23, 0% outliers, 64 ESC-50 clips, one checkpoint). Its repo holds ready-to-submit ICASSP / DCASE / Interspeech / NeurIPS-workshop versions. You **can no longer claim "first to look at AST"**, and the plan's "AST is the bridge model where artifacts will appear first" premise is **refuted** (our pilot agrees).
3. **Pilot (new, ours):** Whisper's encoder shows a textbook Darcet artifact — **~2–3% of tokens at 34–59× median norm, massive activations ~1,100–1,750× median, emerging abruptly at a mid layer, located on silence and on the 30-s zero-padding**. On 30-s all-speech inputs the outliers still land on the quietest frames (**AUROC 0.86** for "low energy → outlier"). HuBERT-Large has a different species: **one massive token per utterance (≈10× norm) that becomes a 110×-uniform attention sink** absorbing 37% of all attention. Post-LN base models (HuBERT-B, WavLM-B+) have **boundary-anchored sinks** instead. AST has none — because its CLS + distillation tokens already absorb up to 45% of patch attention (a built-in register pair; the Zenodo paper missed this).
4. **Plan as written → Interspeech (CORE A) / ICASSP (CORE B) / A\* *workshop*.** For an A\* **main track** (ICML/NeurIPS/ICLR) you need (a) a unifying mechanistic account of *why* the four regimes differ, with causal tests, and (b) one headline practical payoff. The strongest candidate: **training-free padding-free Whisper** via test-time registers (Whisper is the default audio encoder of most audio-LLMs, and an ICASSP-2026 paper needed fine-tuning + distillation to remove the padding).
5. **Deadlines:** ICASSP 2027 (Sep 23) and ICLR 2027 (Sep 25) have **passed**. Realistic targets: **ICML 2027 (abstract Jan 16 / paper Jan 22, 2027)**, ICLR-2027 workshops (~Feb), **Interspeech 2027** (~early Mar), NeurIPS 2027 (~May). Note ICASSP — the plan's primary target — is CORE **B**, not A\*.

---

## 1. What changed since the 2026-07-16 check

### 1.1 Direct overlap (must cite, must differentiate)

| Paper | Date / status | What it does | Threat level |
|---|---|---|---|
| **Imdad, "Silence in the Noise: Probing Attention Artifacts and Register Tokens in AST"** — [Zenodo 20710635](https://zenodo.org/records/20710635), code [real1900/silence-in-the-noise](https://github.com/real1900/silence-in-the-noise) | 2026-06-16, JHU course project; **not indexed by Semantic Scholar**; repo contains ICASSP/DCASE/Interspeech/NeurIPS-workshop drafts | AST only (1 checkpoint, 64 ESC-50 clips): no high-norm outliers at any layer; *trained* registers (n≤16) absorb 11% attention, +0.33 pp ESC-50 (n.s.); main contribution is naive INT8 PTQ of AST. Explicitly lists PaSST/SSAST/BEATs/HTS-AT/AudioMAE/M2D as open. **No speech models, no Whisper, no test-time registers, no sinks analysis.** | **Medium.** Kills the "first look at AST" claim and the AST-as-bridge premise; leaves speech encoders completely open. May land at ICASSP/Interspeech 2027 — cite it as concurrent work. |

### 1.2 Adjacent, new since July (cite; not a scoop)

- **BAT: Better Audio Transformer** (arXiv 2602.16305, Feb 2026) — *hypothesises* attention sinks corrupt EAT/data2vec teacher targets and adds gated attention; never measures sinks. → Motivation for us: the audio SSL community already suspects sinks but nobody has characterised them.
- **Adapting Whisper for Padding-Free Inference** (Oracle, ICASSP 2026, IEEE 11462351) — Whisper "learned to use padding"; removing padding needs fine-tuning + KD (22× speed-up). **No mechanism given.** → Our pilot supplies the mechanism (padding frames host the high-norm register tokens) and suggests a training-free fix.
- **Stride-k Subsampling for Whisper** (2608.30927), **Whisper hallucination SAE steering** (2606.07473), **AURA** (2609.23979), **Calm-Whisper** (2505.12969) — Whisper encoder redundancy / non-speech hallucination; none mention sinks/registers/high-norm tokens.
- **DINO-A** (2608.10659) — DINO for audio; no artifact analysis. **SPARE** (2609.20849) — "register token" in an audio-LLM for reasoning (unrelated). **WnW** (2608.22704) — audio-LLM KV sinks (decoder side).
- **Omni-LLM sinks** (2603.14337), **AV-LLM information flow** (2606.10147) — LLM-decoder side, same differentiation as Llama-AVSR.
- Older evidence to cite: **Yang et al., Interspeech 2020** ("vertical" heads that attend to silence in SSL audio transformers); **XLSR-Transducer** (2407.04439, keeps initial frames as sinks for streaming — consistent with our boundary-sink finding); **Wagner et al. Interspeech 2024** (2406.11022, per-*dimension* outliers in distilled Whisper for PTQ — no token localisation); **Lee et al. 2025, Token Pruning in Audio Transformers** (2504.01690, AST/Audio-MAE attention on low-intensity patches).

### 1.3 Evidence the niche is otherwise empty

- **Citation graph (Semantic Scholar, crawled 2026-09-30):** Darcet et al. 1,018 citers; Jiang et al. (test-time registers) 47; PH-Reg 17; cross-arch reassessment 1; Llama-AVSR sinks 9; Massive Activations 303; When Sink Emerges 221; DVT 55; Online-Register 1; Sink survey 8; StreamingLLM 2,520; Gated Attention 324; + 4 others. → 136 unique audio/speech-related citers; **none characterises artifacts, high-norm tokens or sinks in a pretrained audio encoder.** Jiang et al.'s 47 citers: **0 audio**. List: `verification/s2_audio_citers.txt`.
- **arXiv sweeps** (`verification/arxiv_sweep.txt`): `"high-norm" ∧ (audio∨speech)` → 0; `"outlier tokens" ∧ (audio∨speech)` → 0; `"register neurons"` → only Jiang et al.; `"test-time registers"` → 3, none audio; `"massive activations" ∧ audio` → only Llama-AVSR; `"attention sink" ∧ cs.SD/eess.AS` → 5, all decoder/serving/separation.
- **The 2026 attention-sink survey (2604.10098)** still has audio only on the LLM/AV-LLM side; audio encoders absent.
- **Semantic Scholar keyword searches** (`verification/s2_keyword_search.txt`): nothing relevant beyond the above.

---

## 2. Pilot experiment — does the phenomenon exist? (new evidence)

**Setup (CPU, ~1 h, indicative only):** 13 inputs per model — 6 LibriSpeech utterances padded with 2 s leading + 2 s trailing silence; 3 × (speech | 2 s silence | speech); 3 × trimmed speech touching both boundaries; 1 × white noise. Per layer: per-token L2 norm (pre-final-LN), massive-activation ratio max|h|/median|h|, received attention (head- and query-averaged, × sequence length = "× uniform"), share of attention received by the top-1% tokens. Whisper additionally tested on 30 s of pure speech (no padding), with 5-s digital-silence and noise gaps, across 4 content permutations. Code: `pilot/`.

| Model | LN | High-norm outliers | Massive act. (max/med) | Sink behaviour | Where artifacts sit |
|---|---|---|---|---|---|
| **Whisper-base enc.** | pre | **~2.0% of tokens, 34× median norm, abrupt at L4** | **~1,100×** | 9–17× uniform | **Silence & zero-padding (94–100%)**; with 30-s pure speech → quietest frames, **AUROC 0.86**; positions not fixed across contents (Jaccard ≈ 0, chance) |
| **Whisper-small enc.** | pre | **~3.2%, 59× median, abrupt at L7** | **~1,750×** | 12–18× | Silence & padding (100% of argmax tokens) — **grows with scale** |
| **HuBERT-Large** | pre | **one token per utterance, ≈10× norm, abrupt at L13, persists to L24** | ~490× | **same token → 110× uniform sink; top-1% tokens take 37% of all attention** | Mid-utterance, mid-energy frames (not silence, not boundaries); also in white noise |
| WavLM-Large | pre | mild (≤3.4×) | ~190× | 10–39×, mostly first/last frame | Sequence boundaries |
| HuBERT-Base | post | none (post-LN normalises norms) | ~50× | first → last frame from L5, up to 15–22× | Sequence boundaries, even when speech fills the clip |
| WavLM-Base+ | post | none | ~38× | boundary (L4–7) then content sinks up to 38×, top-1% take 26% | Boundaries, then speech frames |
| AST (AudioSet) | pre | **none** (≤1.8×) — agrees with Imdad | ~50× | patch sinks ≤48×, but **CLS + dist tokens receive up to 45% of patch attention** | Built-in registers (DeiT's two special tokens) |

**What this means scientifically**
- The phenomenon is **real, large, and heterogeneous** across audio encoders — four distinct regimes: (i) *Darcet-style content registers on silence/padding* (Whisper), (ii) *a single LLM-style massive sink token chosen from content* (HuBERT-L), (iii) *boundary-anchored sinks* (post-LN / conv-positional SSL models), (iv) *special tokens act as registers → no artifacts* (AST).
- The plan's RQ2 "silence law" **holds for Whisper and fails for SSL speech models** — that contrast is a finding, not a failure.
- **Post-LN models (HuBERT-Base, WavLM-Base+, wav2vec2-Base) cannot show Darcet high-norm tokens by construction**; the plan's norm-only protocol would call them "artifact-negative" and miss their strong sinks. The protocol must use sinks + massive activations + pre-LN residual norms.
- **Whisper is the killer model:** it is the default audio encoder of most audio-LLMs, its 30-s padding is a well-known inefficiency, and the pilot shows the padding frames are where the model stores its register computation — a mechanistic explanation for why padding-free Whisper breaks.

**Caveats:** 13 clips per model, a dummy LibriSpeech subset, thresholds not pre-registered, head-averaged attention. Treat as existence evidence, not results. Not yet run: BEATs, EAT, Audio-MAE, wav2vec2/XLS-R, Whisper-medium/large-v3.

---

## 3. Is it A\*-worthy? Honest assessment

| Version of the paper | Realistic venue |
|---|---|
| Plan as written (census of 6–8 models + port test-time registers + linear probes + HEAR/SUPERB probes) | Interspeech 2027 (CORE A) / ICASSP (CORE B) / **A\* workshop** (e.g. ICLR/ICML/NeurIPS interpretability or audio workshops). Reviewers will say "vision result transferred". |
| **Re-framed version below** | **ICML 2027 / NeurIPS 2027 main track — plausible, not guaranteed** |

What lifts it to A\* main track:
1. **A unifying mechanism, tested causally.** Explain the four regimes with testable factors: pre- vs post-LN; conv-relative vs sinusoidal vs learned-absolute positions (boundary detectability); presence of special tokens (AST's CLS+dist); training with fixed-length zero-padding (Whisper). Causal tests: (a) register-neuron localisation + activation moving (Jiang et al.), (b) remove/add special tokens in AST at inference, (c) crop/extend padding and move silence in Whisper, (d) **small-scale controlled pretraining** (e.g. tiny HuBERT/Whisper-style encoders on LibriSpeech-960 with vs without padding / CLS / registers / pre-LN) — this is the piece that turns a census into science.
2. **One practical headline with big numbers.** Top candidate: **training-free padding-free Whisper** — insert k test-time registers carrying the outlier activations so short clips run without the 30-s pad; measure WER and speed-up vs the fine-tuned+KD baseline (ICASSP 2026). Secondary: effect on Whisper non-speech hallucination; effect on audio-LLM token pruning/compression (Stride-k, HeadRouter, etc.); INT8 PTQ of Whisper encoder.
3. **Scale trend.** Whisper tiny→large-v3 (and HuBERT/WavLM B→L), showing artifact strength grows with scale (base 34× → small 59× already).

---

## 4. Required changes to `ResearchPlan.md`

1. **§0/§1.6/§1.8:** drop "AST is the bridge model where artifacts appear first"; cite Imdad (2026) and our pilot: AST is artifact-free because of its two special tokens. Keep AST as the *negative control*.
2. **§2 gap statement:** "No work characterises high-norm/sink artifacts in pretrained **speech** encoders (HuBERT, WavLM, Whisper, XLS-R) or self-supervised audio encoders (BEATs, EAT, Audio-MAE); a concurrent preprint reports AST is artifact-free."
3. **§3 Phase A protocol:** add sink metrics (received attention × uniform, top-1% attention share), massive-activation ratio, and pre-LN residual norms; handle post-LN models explicitly; add Whisper with/without padding and 30-s full-speech inputs.
4. **Lead model = Whisper encoder**; add Whisper tiny/base/small/medium/large-v3 scale sweep; add wav2vec2/XLS-R.
5. **Phase C:** first target is test-time registers for **padding-free Whisper**.
6. **Venues/timeline:** ICASSP 2027 is gone. Primary: **ICML 2027 (Jan 22)** if Phases A–C + controlled pretraining are done by mid-January; fallback **Interspeech 2027** (8-page long track) + an A\* workshop. Post a census preprint on arXiv by **~Dec 2026** to timestamp priority (the vocabulary is leaking into audio fast).
7. **Compute:** Whisper-large-v3 / HuBERT-XL inference fits Kaggle T4; small controlled pretraining needs ~100–300 A100-h (budget this now).

## 5. Re-verify before submission

Re-run the arXiv queries in `verification/arxiv_sweep.txt`, re-crawl citers of Darcet et al. / Jiang et al., and watch the `real1900/silence-in-the-noise` repo and Zenodo record for extensions to speech models.
