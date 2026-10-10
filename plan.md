# plan.md — Research Hub · single source of truth

> **Project:** *What do speech transformers do with silence?* — attention sinks, registers and the hidden work of padding in Whisper-family models
> **Status:** Phase 0 (scale-up of pilots) · **Last updated:** 2026-10-11 · **Target:** ICML 2027 (abstract ≈ 2027-01-16, paper ≈ 2027-01-22 — re-verify in Dec) · fallback NeurIPS 2027 / Interspeech 2027
> **Rule:** this file is the only plan. Every claim cites evidence (`[E##]` → [`docs/evidence.md`](docs/evidence.md)) or a run (`results/...`). Old plans are in git-ignored `archive/`.

---

## 0 · TL;DR (read this first)

**Problem.** Whisper — the default ASR model and the audio encoder inside most audio-LLMs — always pads input to 30 s. Removing the padding makes it collapse (our pilot: WER 5→373 % on Whisper-base, 2.6→214 % small, 2.0→428 % large-v3-turbo, mostly runaway repetition) [E22, E23]. Everyone works around this with *trained* fixes (hush word, WhisperKit, ACFT, FT+KD) [E08, E09]; **nobody knows mechanistically what the 25 s of silence is used for.**

**Why now / why us.** Attention sinks, massive activations and registers are the most active mechanism topic at A* venues in 2026 (ICML'26 ≥ 24 papers; ICLR'27 74 submissions) with *competing theories* [E03, E04, E06, E13, E16, E18, E19] — yet **no work studies them in speech encoders, silence or padding** (verified across ICLR'27 submissions, ICML'26, Interspeech'26, CVPR/ECCV/ACL'26, arXiv to 2026-10-10 → [`docs/scoop_check_2026-10-10.md`](docs/scoop_check_2026-10-10.md)). Speech offers what text and images cannot: **exactly controllable "nothing"** (digital silence, constant padding frames, stationary noise) that lets us dissociate energy, redundancy and position.

**What we found already (pilots, n = 100 utts × 3 models; provisional).**
1. Whisper's encoder forms textbook massive-activation "register" frames on padding (Ψ = 1,122, |x| = 501 in base) — **but they are dispensable for ASR** (masking/removing them: no WER change).
2. What Whisper needs from padding is **silence, used two ways**: the decoder needs silent encoder states as *end-of-speech evidence*, and the encoder's speech frames attend to a handful of **position-anchored attention-sink frames at the start/end of the 30-s window** (large-v3-turbo: 16 top-attended cached frames → WER 2.6 % vs 17.5 % for 16 random frames).
3. A **training-free "SilenceCache"** (precompute silence K/V + silence tail once; encode only the speech) matches or beats full padding (base 5.1 vs 5.3 %; small 2.6 vs 2.6 %; turbo 1.7 vs 2.0 %) at **≈0.16× encoder FLOPs** for 2–8 s clips.

**The paper (one sentence).** *Speech transformers turn silence into computation: Whisper's padding hosts position-anchored attention sinks and end-of-speech evidence — not its high-norm registers — and caching a few silence sinks removes padding for free.*

**Contributions (target).** (C1) first census of registers / massive activations / sinks across speech & audio encoders, with natural experiments (pre-/post-LN; special tokens; padding visible vs masked vs absent); (C2) controlled acoustic tests that adjudicate 2026 theories of *where* sinks form; (C3) causal dissociation **sinks ≠ registers** in production speech models (encoder self-attention and decoder cross-attention); (C4) SilenceCache: training-free padding-free Whisper with wall-clock speed-ups, across sizes/languages/domains, vs trained baselines.

---

## 1 · Dashboard

### 1.1 Deadlines & venues
| Venue | Rank | Deadline (as known) | Fit | Decision |
|---|---|---|---|---|
| **ICML 2027** | A* | abstract ≈ 2027-01-16, paper ≈ 2027-01-22 (session-1 check; re-verify) | science-of-DL + efficiency | **primary** |
| NeurIPS 2027 | A* | ≈ May 2027 | same; more time for LALM part | fallback (extended) |
| Interspeech 2027 | A (CORE) | 2027-02-09 (verified 2026-10-10) | Whisper-specific short version | fallback; *no dual submission* |
| ICLR 2027 workshops | A* workshop | ≈ Feb 2027 | early visibility (non-archival) | optional |
Go/no-go for ICML: **2026-12-20** (see §9 kill criteria).

### 1.2 Phase tracker
| Phase | Window | Goal | Exit criterion | Status |
|---|---|---|---|---|
| P0 Pilot | 10-10 → 10-11 | de-risk core premise | effect on ≥ 2 models | ✅ done (3 models) |
| **P1 Scale-up & census** | 10-12 → 10-31 | replicate pilots at full scale; census of 18+ checkpoints | E1–E3 complete, CIs | ⏳ next |
| P2 Mechanism | 11-01 → 11-28 | theory tests, sink-vs-register dissociation, decoder anatomy | E4–E8 complete | ☐ |
| P3 Method & benchmarks | 11-15 → 12-20 | SilenceCache vs baselines, wall-clock, languages, LALM | E9–E11 complete | ☐ |
| P4 Writing | 12-01 → 01-15 | full paper in `paper/` | internal review ×2 | ☐ |
| P5 Submit | 01-16 → 01-22 | ICML submission + code release plan | submitted | ☐ |

### 1.3 Results so far (keep updated)
| Run | Model(s) | n | Key numbers | File |
|---|---|---|---|---|
| Census v0 (session 1) | 7 encoders | 13 inputs | Whisper-base 34× norm, ~2 % tokens; small 59×; HuBERT-L single massive sink; post-LN Base: sinks w/o norm outliers; AST: CLS+DIST absorb ≤ 45 % | `experiments/census_2026-09/` |
| Pilot v1 padding | Whisper-base | 100 | pad30 5.39 · nopad 385.5 · pad1 31.2 · pad5 5.94 · reg-transplant 315 · reg-mask 5.39 | `experiments/pilot_padding/results/whisper-base_n100_r4.json` |
| Pilot v2 SilenceCache | base / small / large-v3-turbo | 100 each | kv+tail 5.12 / 2.64 / 1.71 vs pad30 5.28 / 2.56 / 2.02 at 0.157× FLOPs; turbo kvTOP16 2.64 vs kvRND16 17.53 | `experiments/pilot_padding/results/*_padkv_n100.json` |
| Pilot v3 non-speech | Whisper-base · large-v3-turbo | 200 ESC-50 each | % clips with any output (mean words): base pad30 1.0 (0.80) · nopad 8.5 (1.37) · nopad+tail 5.0 (0.06); **turbo pad30 78.5 (1.10) · nopad 98.0 (16.27) · nopad+tail 100 (1.32) · tile30 94.0 (2.81)** → silence tail cuts runaway length but not the occurrence of short non-speech hallucinations; E12 stays exploratory | `experiments/pilot_padding/results/*_esc50_n200.json` |

### 1.4 Top risks (details §9)
1. Pilot effects shrink at scale / other datasets → P1 replication is the gate.
2. "Sinks ≠ registers" may be an artefact of *which* frames we masked (norm vs attention sets overlap in base) → measure overlap per layer before claiming.
3. SilenceCache seen as "a trick" → must be derived from, and explained by, the mechanism; broad evaluation; wall-clock.
4. Scoop by sink groups moving to audio (KAIST MM, RegCache, Triage) → weekly watch (§10); arXiv preprint after P2 if needed.

---

## 2 · Research question & novelty

### 2.1 Questions
- **RQ1 — Where?** Which speech/audio encoders form (a) high-norm register frames, (b) massive activations, (c) attention sinks; where do they sit (silence, padding, speech, absolute positions); how do normalisation, special tokens, objective, scale and **padding regime** change this?
- **RQ2 — Why there?** Which 2026 theory predicts their location and emergence layer when we control energy, redundancy, position and duration of "nothing" in the input?
- **RQ3 — What for?** Which of these structures are *functionally necessary* — in encoder self-attention and in decoder cross-attention — and for what (termination, alignment, global information)?
- **RQ4 — Use it.** Can cached silence sinks make Whisper padding-free *without training*, across sizes, languages, domains and Whisper-derived audio-LLMs, and how does that compare with trained alternatives?
- **RQ5 — Beyond (stretch).** Do LALM encoders that never see padding relocate sinks into speech/silence? Does sink placement matter for non-speech hallucination and audio-token pruning?

### 2.2 What is new vs closest work (verified)
| Closest work | What it does | What it does *not* do (our gap) |
|---|---|---|
| Jiang et al. NeurIPS'25 test-time registers [E01]; RegCache ECCV'26 [E02] | registers/cached KV in **vision** encoders | speech; silence; decoder cross-attention; padding |
| Anatomy ICML'26 [E03]; Structural Origin ICML'26 [E04]; ME-layer ICML'26 [E19]; Compression valleys ICLR'26 [E18] | mechanisms of sinks/massive activations in **causal LLMs** (BOS anchor) | bidirectional encoders without BOS; position-anchored sinks at a window *end*; production-model dissociation |
| Mind the Spike (ICLR'27 sub.) [E06]; To Sink or Not ICLR'26 [E05] | visual spikes / ViT sinks inside **LVLMs** | audio; silence; padding |
| Llama-AVSR sinks ICASSP'26 [E20]; AV-LLM hubs ICML'26 [E21]; Triage [E07] | sinks in the **LLM** of audio(-visual) LLMs | the audio **encoder** itself |
| WhisperFlow hush word [E08]; WhisperKit [E09]; FUTO ACFT; ICASSP'26 FT+KD; NPUsper [E10]; stride-k [E12] | engineering for short-input / efficient Whisper | mechanism; training-free padding removal that preserves the sinks |
| Imdad (Zenodo 2026, non-peer-reviewed) | AST has no artifacts | why (CLS+DIST act as registers — our census); speech encoders |

**Novelty statement (what reviewers should remember).** First mechanistic account of how speech transformers use silence; first production-model evidence that attention sinks and high-norm registers dissociate functionally; first training-free padding-free Whisper derived from that mechanism.

---

## 3 · Hypotheses & predictions (pre-registered; edit only via §11 decision log)

| ID | Hypothesis | Source theory | Prediction in speech | Test (exp.) | Confirm / kill |
|---|---|---|---|---|---|
| H1 | Norm-type matters: pre-LN → sinks **and** massive activations; post-LN → sinks without massive activations | [E03] | Whisper & pre-LN SSL-Large: Ψ ≥ 1000 & sinks; post-LN Base SSL (HuBERT-B, wav2vec2-B, WavLM-B+) & wav2vec2-large (post-LN Large): sinks, Ψ < 1000 | E1 | confirm if split holds in ≥ 80 % of checkpoints |
| H2 | Sinks need a "nothing" anchor (default state) | [E13] | when silence/padding is absent (speech-only 30 s), sinks move to the lowest-information speech frames or to absolute positions; sink strength ↓ | E2, E5 | kill if sink location is unchanged by removing silence |
| H3a | Redundancy rule (Darcet) | [E11] | sinks/registers on frames most similar to neighbours (constant padding, tones) regardless of energy | E5 factorial | — |
| H3b | Least-shared / smallest rule | [E06] | sinks on low-norm / least-shared frames (silence, quiet noise), not on loud constant tones | E5 factorial | the factorial picks one of H3a/H3b/H3c |
| H3c | Self-sinking blocks | [E03, E04] | a block of identical frames that attends to itself becomes a sink; masking any frame to self-only creates a sink | E5, E6 | — |
| H3d | Position anchoring | pilots [E23] | in Whisper, sinks sit at fixed absolute positions (0, ~1450–1499) whenever those positions hold silence | E5 position shift | kill if sinks follow content when silence is moved |
| H4 | Early emergence when "nothing" is identifiable from the input | [E02] | Whisper registers emerge earlier with constant padding than with noisy silence; speech-only inputs → later/weaker | E2 | — |
| H5 | **Sinks ≠ registers** | [E03] + pilots | masking top-*attention* frames hurts ASR; masking top-*norm* frames does not | E7 | kill if both or neither hurt equally |
| H6 | Decoder needs silence as end-of-speech evidence | [E10] + pilots | removing silent encoder states → repetition loops; restoring a content-free silence tail restores EOT; a short tail (≤ 1–2 s) suffices | E8 | kill if tail length ≫ 5 s is required |
| H7 | Dependence on encoder sinks grows with scale | pilots | kvTOP-vs-kvRND gap: tiny ≈ base ≈ small < medium < large-v3/turbo | E3, E9 | — |
| H8 | SilenceCache is near-lossless and training-free | pilots | ΔWER ≤ +0.3 abs (or ≤ 5 % rel) vs pad30 on LS clean/other, TED-LIUM, FLEURS (5+ langs); ≥ 3× wall-clock encoder speed-up for ≤ 10 s clips | E9, E10 | §9 kill criterion K2 |
| H9 (stretch) | LALM encoders with masked padding relocate sinks into speech | [E12] | Qwen2-Audio/AF3 encoders: sinks on in-utterance silence/low-energy frames; Whisper-large-v3: on padding | E11 | — |

---

## 4 · Experiments (detailed map)

Conventions: all inference-only unless stated; fp16 on GPU; seeds fixed; WER with Whisper's English normaliser (`tokenizer.normalize`) and `jiwer`; 95 % CIs by bootstrap over utterances (1,000 resamples); "hallucination" = (hyp words > ref words + 3) or a ≥ 2× repeated 3-gram (see `experiments/pilot_padding/whisper_regs.py::is_halluc`).

| # | Experiment | Data | Models | Compute | Status | Output |
|---|---|---|---|---|---|---|
| E1 | **Census** of registers / massive activations / sinks per layer (norm ratio, Ψ with |x|>100 floor, sink share, entropy H(X), anisotropy p1) and *location* (silence / padding / speech / absolute position) | LS test-clean 200 utts × {speech+2 s sil, speech-only, 30-s speech}, ESC-50 200 | ~18 ckpts (§5) | 3050: Whisper ≤ turbo, SSL-B/L; T4: large-v3, XLS-R-1B, LALM encoders | ☐ (v0 done on 7) | `results/census/*.json`, Fig.1 |
| E2 | **Padding/silence dose–response** for Whisper: padding length 0–25 s; silence type (digital 0, −60 dB noise, −40 dB noise, constant tone); speech-only 30 s | LS test-clean 500 | tiny→large-v3, turbo | ≈ 6 GPU-h | ☐ | Fig.2 |
| E3 | **Replicate pilot v2 at scale** (all SilenceCache conditions, TOP/RND k ∈ {1,2,4,8,16,32,64,256}) | LS clean+other full (5.5k), TED-LIUM 3 short-form | tiny, base, small, medium, large-v3, turbo | ≈ 20 GPU-h (T4) | ☐ | Tab.2, Fig.3 |
| E4 | **Register-vs-sink overlap** per layer (Jaccard of top-norm vs top-attention sets; ME-layer analysis [E19]; register-neuron search [E01]) | as E1 | Whisper ×6, HuBERT-L, WavLM-L | ≈ 3 GPU-h | ☐ | Fig.4 |
| E5 | **Location factorial** (energy × redundancy × position × duration): insert 1 s blocks of {digital silence, quiet noise, loud constant tone, loud noise} at {start, middle, end} of speech; *move* silence within the 30-s window; region-scaling à la [E06] | LS 300 | Whisper ×4, HuBERT-L | ≈ 5 GPU-h | ☐ | Tab.3 (theory scoreboard) |
| E6 | **Mask-to-self intervention** [E04] and trigger-direction removal [E06] in bidirectional encoders: does forcing a frame to self-attend create a sink? | LS 200 | Whisper-small, turbo | ≈ 2 GPU-h | ☐ | Fig.5 |
| E7 | **Causal necessity**: mask top-k attention-sink frames vs top-k norm frames vs random (encoder self-attn; per layer band) | LS 1k | Whisper ×6 | ≈ 6 GPU-h | partially (pilot) | Tab.4 |
| E8 | **Decoder anatomy**: cross-attention mass on silence per step (esp. at EOT, punctuation); tail-length dose–response (0.1–25 s, nearest-first vs top-attended); NPUsper-style backward-shift signature | LS 1k | Whisper ×6 | ≈ 4 GPU-h | ☐ | Fig.6 |
| E9 | **SilenceCache benchmark**: WER + wall-clock (encoder ms, end-to-end RTF, peak memory) vs pad30, nopad, padK, hush word (re-trained per model, [E08] recipe), FUTO ACFT ckpts (tiny–small), stride-k [E12] (and combined) | LS clean/other, TED-LIUM 3, FLEURS (en, de, fr, es, hi, bn, zh), CV-17 subset; short commands (≤ 3 s) | tiny→large-v3, turbo, distil-large-v3 | ≈ 40 GPU-h (T4); hush-word training ≈ 2 h/model | ☐ | Tab.1 (main), Fig.7 |
| E10 | **Robustness**: noisy speech (MUSAN/CHiME-like SNR sweep), long-form chunking, streaming (LocalAgreement), beam search vs greedy, temperature fallback | LS + MUSAN, TED-LIUM long-form | base, small, turbo | ≈ 10 GPU-h | ☐ | App. |
| E11 | **LALM natural experiment** (stretch): sink census in audio towers of Qwen2-Audio-7B (masked padding), Audio Flamingo 3 (masked), Qwen2.5-Omni-3B (windows), Voxtral-mini (check); SilenceCache for LALMs that pad | LS 200, MMAU-mini 200 | 3–4 LALMs | Kaggle 2×T4 (8-bit) ≈ 10 GPU-h | ☐ | §6 of paper |
| E12 | **Non-speech hallucination** (exploratory): silence tail / sink restoration on ESC-50, MUSAN, UrbanSound; baselines Calm-Whisper, SAE steering [2606.07473] | ESC-50 2000, MUSAN | base, small, turbo, large-v3 | ≈ 4 GPU-h | pilot running | App. or cut |

**Sequencing:** E3 → E1/E4 → E7/E8 → E5/E6 → E9/E10 → E11/E12. E3 is the gate (if it fails, see §9).

---

## 5 · Models & data

### 5.1 Model grid (HF ids; "fits" = local RTX-3050 4 GB fp16 / Kaggle T4 16 GB)
| Family | Checkpoints | LN | Positions | Padding regime | Fits |
|---|---|---|---|---|---|
| Whisper | tiny, base, small, medium, large-v3, large-v3-turbo (`openai/whisper-*`); distil-large-v3 | pre | sinusoidal (enc) | visible zero-pad to 30 s | 3050: ≤ medium, turbo; T4: large-v3 |
| Moonshine | `UsefulSensors/moonshine-tiny/base` | pre | RoPE | **no padding** (variable length) | 3050 |
| SSL Base (post-LN) | `facebook/hubert-base-ls960`, `facebook/wav2vec2-base`, `microsoft/wavlm-base-plus`, `facebook/data2vec-audio-base` | post | conv-relative | batch padding only | 3050 |
| SSL Large (pre-LN) | `facebook/hubert-large-ll60k`, `microsoft/wavlm-large`, `facebook/wav2vec2-large-lv60`, `facebook/wav2vec2-xls-r-1b` | pre | conv-relative | batch padding only | 3050 (≤ L); T4 (1B) |
| SSL Large (post-LN) | `facebook/wav2vec2-large` (LS-960, post-LN) | post | conv-relative | — | 3050 |
| Audio (patch) | `MIT/ast-finetuned-audioset-10-10-0.4593` (CLS+DIST), BEATs (iter3), EAT-base/large, Dasheng-base (MAE) | pre | learned / 2-D | fixed 10.24 s | 3050 |
| LALM audio towers | Qwen2-Audio-7B (masked), Audio Flamingo 3 (masked), Qwen2.5-Omni-3B (windows), Voxtral-mini-3B (to check) | pre | sinusoidal | as noted | T4 (encoder-only fits 3050) |

### 5.2 Data
LibriSpeech test-clean/other (local: `data/LibriSpeech/test-clean`), TED-LIUM 3 (short-form test), FLEURS (7 langs incl. Bengali), Common Voice 17 subset, ESC-50 (local: `data/ESC-50-master`), MUSAN noise, short-command set (TBD: SLURP or Speech Commands). Everything lives in git-ignored `data/`; download commands go in `experiments/README.md`.

---

## 6 · Protocol & metric definitions (fixed; changes go in §11)

- **Hidden states:** recorded by our own forward (`HandEncoder`, matches HF to 7e-4) *before* any final LayerNorm (`ln_post`); HF `hidden_states` conventions differ by model/version — never mix.
- **Norm ratio** = max token ℓ2 / median token ℓ2 per layer. **High-norm/register frame:** ratio > 10 (report also the bimodality, Darcet-style).
- **Massive activation (Sun et al. [E14]):** |x| > 100 AND ≥ 1,000× median |x| of the hidden state; we report Ψ = max|x| / median|x| (not the within-token φ, which flags almost everything [E06]).
- **Attention-sink frame:** received attention (mean over heads and queries, × sequence length = "× uniform"); sink if share ≥ 0.3 of total attention in ≥ 1 deep layer (Gu et al. via [E06]) — also report top-k by received attention.
- **Entropy / anisotropy:** matrix-based entropy H(X) and p1 = σ1²/‖X‖²_F per layer [E18].
- **Location labels:** speech / in-utterance silence (energy VAD < −40 dBFS) / padding / absolute position bins.
- **ASR:** greedy unless stated; Whisper normaliser; WER + hallucination rate; bootstrap CIs; paired tests vs pad30.
- **Cost:** analytic encoder FLOPs (`whisper_padkv.py::enc_flops`) **and** wall-clock (CUDA events, batch 1 and 16, fp16, T4 + RTX-3050), peak memory.
- **SilenceCache definition:** encode a 30-s silence input once → cache per-layer K/V (positions [T,1500)) and final states (tail). Online: encode only the T speech frames; each speech query attends to its own keys plus cached keys (all, or top-k by silence-run received attention); decoder cross-attends to speech states + cached tail (all or first m frames).

---

## 7 · Compute & budget
- Local RTX-3050 4 GB (venv: `.venv`, torch 2.11 + CUDA 12.8, transformers 5.19): Whisper ≤ turbo, SSL ≤ Large, AST/BEATs/EAT.
- Kaggle free 2×T4 (≈ 30 h/week): large-v3, XLS-R-1B, LALMs, full benchmarks. Plan ≈ 110 GPU-h total over P1–P3 → feasible on free tiers in ~6 weeks; budget USD 50–100 rental for a final A100 sweep (wall-clock on server GPU) if needed.

---

## 8 · Paper map (`paper/`, supervisor template, REVTeX two-column)
| Section | Content | Evidence / runs |
|---|---|---|
| 1 Intro | padding problem; silence as computation; contributions C1–C4 | E22, E23; Fig.1 |
| 2a Background | Whisper; sinks, massive activations, registers (definitions) | E03, E14, E15 |
| 2b Related | registers (vision); sink theory 2026; LVLM/LALM sinks; efficient Whisper | E01–E21 |
| 3 Method | measurement protocol; interventions (masking, KV caching, tail); SilenceCache algorithm; FLOPs | §6 |
| 4 Evaluation | census (E1), theory scoreboard (E5/E6), dissociation (E4/E7), decoder anatomy (E8), SilenceCache benchmark (E9/E10), LALMs (E11) | Tab.1–4, Fig.2–7 |
| 5 Discussion | what silence buys; limits; implications for padding-trained models | — |
| 6 Conclusion | — | — |
| 7 Appendix | notation, FLOP derivation, implementation, extra results | — |

---

## 9 · Risks, kill & pivot criteria
| ID | Risk | Signal | Mitigation / pivot |
|---|---|---|---|
| K1 | Pilot effects do not replicate at scale (E3) | kv+tail ΔWER > +1 abs on LS-clean for ≥ 2 sizes | pivot to the analysis paper (C1–C3) for Interspeech/ICASSP; SilenceCache becomes a negative result |
| K2 | SilenceCache wall-clock gain small (attention overheads, kernels) | < 2× encoder speed-up at ≤ 10 s on T4 | report FLOPs + memory; use top-k cache (k ≤ 16) which is ~nopad cost; implement fused SDPA with extra KV |
| K3 | "sinks ≠ registers" collapses after overlap analysis (E4) | top-norm and top-attention sets coincide in the layers that matter | reframe as "early sinks (pre-emergence) vs late registers"; still a C3 result |
| K4 | Theories make indistinguishable predictions | E5 scoreboard ties | keep E5 as descriptive; emphasise position anchoring (H3d) |
| K5 | Scoop | new arXiv on audio sinks/registers or padding-free Whisper | immediately post an arXiv preprint of P1–P2 results; differentiate on SilenceCache + dissociation |
| K6 | Reviewer: "Whisper-specific" | — | E1 census (18 ckpts), Moonshine (no padding) and LALM regimes (E11) show generality; frame as *padding-trained transformers* |

---

## 10 · Scoop monitoring (weekly, Mondays; log in §12)
- arXiv: `(abs:whisper OR abs:speech OR abs:audio) AND (abs:"attention sink" OR abs:"massive activation" OR abs:register OR abs:"high-norm" OR abs:padding)` sorted by date (script: `scripts/` → add `arxiv_watch.py` in P1).
- OpenReview ICLR 2027 (reviews out ≈ Nov) — recheck the 6 closest submissions in `docs/scoop_logs/2026-10-10/openreview_iclr2027_icml2026.md`.
- People: KAIST MM (J.S. Chung), Llama-AVSR (Cappellazzo/Petridis), RegCache (J. Lee, POSTECH), test-time registers (Gandelsman), Triage authors, Imdad (JHU), WhisperFlow (F. X. Lin), Argmax/WhisperKit.

---

## 11 · Decision log (newest first; never delete)
| Date | Decision | Why | Evidence |
|---|---|---|---|
| 2026-10-11 | **Reframe from "do audio transformers need registers?" to "what speech transformers do with silence" (sinks vs registers; SilenceCache)** | pilots: registers dispensable; silence sinks + decoder tail necessary; training-free method works on 3 sizes | E22, E23 |
| 2026-10-11 | Ground hypotheses in 2026 A* theory (ICML'26 ×4, ICLR'26 ×2, CVPR'26, ECCV'26, ACL'26) — Darcet 2023 only as origin | supervisor: avoid ideas resting on ≤ 2025 papers | evidence.md |
| 2026-10-10 | Gap verified open (ICLR'27 42k submissions, ICML'26 6.3k, Interspeech'26, arXiv) | scoop check | scoop_check_2026-10-10.md |
| 2026-10-10 | Old plans (`ResearchPlan*.md`, gap reports, lit-review scripts, pptx, audio) moved to git-ignored `archive/`; git history squashed (old history in `archive/pre-overhaul-2026-10-10.bundle`) | user request: clean repo | — |

---

## 12 · Lab notebook (dated; newest first)
**2026-10-11 (b)** — Paper draft written in the supervisor template (`paper/`, 10 pages, compiles with pdflatex+bibtex; figures generated by `experiments/pilot_padding/make_figures.py`). Contains pilot numbers only, clearly labelled; full-scale experiments described as planned. ESC-50 on turbo: silence tail shortens runaway outputs (16.3 → 1.3 words) but not the occurrence of short non-speech hallucinations (78.5–100 % clips with output). Git: history squash was blocked by the tool's safety check → restructure committed as a normal commit; squash pending user approval (see README).
**2026-10-11** — Pilot v2 on small and large-v3-turbo done (see §1.3). Turbo needs *full* padding (pad5 WER 35 %) but 16 cached top-attended silence frames recover 2.6 %; random 16 → 17.5 %. Silence-only sinks at positions [0, 1496, 1499, 1487, 1455, …]. Masking top-norm frames from the emergence block (20/32) → no harm. ESC-50 probe (base): weak. *Next:* overlap of top-norm vs top-attention per layer; tail-length dose–response; scale to full LS.
**2026-10-10** — Venv + CUDA on RTX-3050 working. Hand encoder validated vs HF. Pilot v1: nopad catastrophic (385 %); 5 s padding ≈ full in base; register transplant useless; register masking harmless. Downloaded 22 new papers (TeX) and read 21 in full → `docs/evidence.md`.

---

## 13 · Task board (owner initials in brackets; tick when done)
**P1 (by 2026-10-31)**
- [ ] Refactor `experiments/pilot_padding` into `experiments/silence/` package (shared encoder, cache, metrics, bootstrap) [ ]
- [ ] E3 full LS clean+other, 6 Whisper sizes (Kaggle notebook) [ ]
- [ ] E4 per-layer overlap norm-vs-attention + register-neuron search [ ]
- [ ] E1 census on 18 checkpoints (extend `census_2026-09/pilot.py`) [ ]
- [ ] E8 tail-length dose–response [ ]
- [ ] `scripts/arxiv_watch.py` weekly scoop query [ ]
**P2 (by 2026-11-28)** E5, E6, E7, E8 full; theory scoreboard table.
**P3 (by 2026-12-20)** E9 benchmark incl. hush-word re-training and ACFT; wall-clock; E10; E11 if time.
**P4/P5** ✅ first draft in `paper/` (2026-10-11, pilot-only); next: fill author block, update with E3/E1 results, internal reviews, submission.

---

## 14 · Reading list (★ = read in full this session; see evidence IDs)
★ E01 test-time registers · ★ E02 RegCache · ★ E03 Anatomy · ★ E04 Structural origin · ★ E05 To Sink or Not · ★ E06 Mind the Spike · ★ E07 Triage · ★ E08 WhisperFlow · ★ E09 WhisperKit (relevant §) · ★ E10 NPUsper · ★ E12 Stride-k · ★ E13 Sinks provably necessary · ★ E16 LaSt-ViT · ★ E18 Compression valleys · ★ E19 ME layer · ★ E20/E21 audio-LLM sinks.
To read next: 2601.22966 (attention & residual sinks, outlier-driven rescaling), 2603.17771 (gradient sinks), 2605.09313 (DiT sinks causal), 2604.10697 (SinkProbe), 2606.07473 (Whisper SAE hallucination), 2505.12969 (Calm-Whisper), 2508.12301 (CarelessWhisper/WhisperRT), FUTO ACFT model cards, ICASSP'26 *Adapting Whisper for Padding-Free Inference* (IEEE 11462351; paywalled — get via library).
All IDs: [`papers/manifest.txt`](papers/manifest.txt) (re-download with `python scripts/download_arxiv_papers.py --manifest papers/manifest.txt`).

---

## 15 · Glossary
**Register (frame):** a token whose hidden-state norm is ≫ median (Darcet); here "high-norm frame". **Massive activation:** single-channel value > 100 and ≥ 1000× median (Sun). **Attention sink:** frame receiving disproportionate attention. **SilenceCache:** our training-free scheme (cached silence K/V + tail). **pad30 / nopad / padK:** zero-pad to 30 s / no padding / K seconds of padding. **TOPk / RNDk:** k most-attended / k random cached silence frames. **Emergence block:** first block where the norm ratio exceeds 10. **Tail:** cached encoder outputs of silence at positions [T, 1500).
