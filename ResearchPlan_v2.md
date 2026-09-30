# Research Plan v2 — Registers, Silence, and the Emergence of High-Norm Tokens in Audio Transformers

**Supersedes:** `ResearchPlan.md` (2026-07-15). Evidence base: `GapVerification_2026-09-30.md` (2026-09-30).
**Status:** proposal draft for supervisor review · **Version date:** 2026-09-30
**Ambition:** A\* main conference (ICML / NeurIPS / ICLR / ACM MM). Floor: A\* workshop. Interspeech/TASLP as the guaranteed landing zone.
**Scheduling stance:** no fixed deadline. Phases are gated by results, not by a calendar. Quality of the mechanism story is the objective function.
**Compute envelope:** inference-dominated. Free Kaggle T4/P100 (30 h/week) covers ~90%; local RTX 4050 (6 GB) for development; ~20–40 A100-hours for the one optional training phase.

---

## 0. The pitch, in one paragraph

Large transformers grow a handful of tokens whose residual-stream norms explode by an order of magnitude. These "artifact" or "register" tokens corrupt attention maps, break dense prediction, and dominate quantization error. Three incompatible explanations are on the table — the model **recycles low-information tokens** as scratch space (Darcet et al., ICLR 2024), sinks are **positional attention no-ops** anchored to whatever index is reliably present (StreamingLLM; Gu et al.; Barbero et al.), or high-norm tokens are a **weight-intrinsic optimization artifact** expressing dominant singular directions and regulating gradient flow (Sun et al.; Wang et al.; the 2026 gradient-sink work). Nobody has been able to separate them, because in images "empty sky" is still content, and in text the BOS token's emptiness is perfectly confounded with its position. **Audio breaks the confound.** Whisper's encoder is contractually fed a 30-second mel window, so a 3-second clip arrives with 27 seconds of *exactly constant, information-free* input whose length and position we control freely; audio lets us fill that region with high-energy-but-structureless noise, or with a looped 100 ms fragment that is locally rich and temporally redundant, dissociating energy from information from redundancy; frame-based speech encoders accept arbitrary sequence lengths on a fixed checkpoint; Whisper ships a **scale ladder at fixed data and fixed objective** that vision has never had; and one Whisper checkpoint contains a bidirectional encoder and a causal decoder trained together. We use pretrained audio and speech encoders as the instrument to adjudicate the three accounts, and we cash the mechanism out on a live, expensive failure: Whisper emits words on 61.9% of pure room-tone clips, and every published remedy is decoder-side or data-side. We argue the encoder's non-speech frames are where the model has parked its registers, and that a training-free test-time register relocates them.

**The paper's claim is about transformers, not about audio. Audio is the apparatus.**

---

## 1. What changed from v1, and why

| v1 assumption | Status after the 2026-09-30 sweep | Consequence for v2 |
|---|---|---|
| "No paper analyzes artifact tokens in *any* pretrained audio encoder." | **False.** One Zenodo preprint (Imdad, JHU, 2026-06-16, DOI `10.5281/zenodo.20710635`) measures AST. Invisible to structured search: not on arXiv, **not indexed in Semantic Scholar**, 0 citations. | Cite it; reproduce it correctly; scope our claim to *multi-model, multi-tokenisation, multi-scale*. See §8.1. |
| "AST is where we'll find artifacts first, because it starts from DeiT weights." | **Backwards.** Darcet's artifacts appear in DeiT-III/DINOv2 **L/H/g**, largely not Base. AST is ViT-**Base** trained on ~5k h. The Zenodo null result is what the literature predicts. | Grid moves up the scale axis. Base models become the **negative-control arm**, not the primary. |
| Model grid: 6–8 checkpoints, almost all ~90M. | Below the emergence regime. | 18–22 checkpoints spanning 20M → 2B, with three within-family scale ladders. |
| "All three RQ1 outcomes are publishable." | True for A/workshop tier. **False for A\* main track**: "we checked Base-scale models for a Large-scale phenomenon and found nothing" is not an A\* result. | The paper is reorganised around a *mechanism adjudication* that yields a result on every branch, plus an application that pays off on the positive branch. |
| Downstream story = "dense tasks improve, like in vision." | Too weak to headline; the Zenodo paper already predicts it in its limitations. | Headline application becomes **Whisper non-speech hallucination**, where all 2026 baselines are decoder-side. SED becomes secondary. |
| Venues: ICASSP 2027 → Interspeech 2027 → TASLP. | ICASSP is CORE **B**, Interspeech **A**. Neither is A\*. | Venue ladder rewritten, §11. |
| Norm measurement convention left implicit. | The one prior audio measurement got this wrong (post-LayerNorm). | Measurement protocol promoted to a first-class, pre-registered section (§4). |

---

## 2. Three accounts, and the predictions that separate them

This table is the spine of the paper. Each row is an experiment we can run only because the modality is audio.

Let **H-A** = token recycling / redundancy (registers go where local content is uninformative).
Let **H-B** = positional attention-no-op sink (registers go to a reliable *index*, near-independent of content; favoured by causal masking and softmax normalisation).
Let **H-C** = weight-intrinsic optimization artifact (a fixed high-norm *direction* set by the layer maps; token identity secondary; emerges with training compute; the gradient-regulation variant requires causal masking).

| # | Experiment (audio-only) | H-A predicts | H-B predicts | H-C predicts |
|---|---|---|---|---|
| **E1** | Whisper 30 s window: place *L* s of speech at start / middle / end, zero-pad the rest | high-norm tokens sit **in the pad**, and *move* when the pad moves | high-norm tokens sit at a **fixed frame index** regardless of where the pad is | token identity unstable across inputs; the high-norm **direction** is constant |
| **E2** | Fill the non-speech region with: digital zero · low white noise · loud white noise · pure tone · pink noise · looped 100 ms real fragment | prefers **low-information** fills, incl. loud-noise and looped fills if *redundancy* (not energy) is what matters | indifferent to fill | indifferent to fill |
| **E3** | Sweep pad duration 1 s → 29 s | register mass / count rises **monotonically** with pad extent | flat | flat |
| **E4** | Whisper scale ladder (tiny→large-v3) at **fixed data and objective**; plus HuBERT B/L/XL, WavLM B+/L, XLS-R 0.3/1/2B | emerges once spare capacity exists | emerges with depth | emerges with **training compute** |
| **E5** | Whisper **encoder (bidirectional) vs decoder (causal)**, same checkpoint | present in both | concentrated in the **causal decoder** | gradient-sink variant: decoder only |
| **E6** | **Post-LN vs pre-LN** at matched scale (HuBERT/WavLM Base are post-LN; Large are pre-LN) | orthogonal | orthogonal | **post-LN renormalises the residual stream every block → no accumulation → artifact-negative by construction** |
| **E7** | Axis asymmetry: band-limited and time-masked inputs on patch models | registers follow the *uninformative axis* (e.g. empty high-mel bands) | n/a | n/a |
| **E8** | Frame models, same checkpoint, 1 s → 60 s inputs | count scales with number of redundant frames | small fixed set near the edges | fixed |
| **E9** | Port `FindRegisterNeurons`; causally *move* an outlier; controls = random neuron, zero-without-move | a sparse causal neuron set exists and relocates outliers | same mechanism, but keyed to position | neurons are the readout of the singular direction |
| **E10** | Singular-defect test: are audio high-norm directions predicted by dominant singular vectors of per-layer linear approximations? | weak | weak | **strong and input-independent** |

**Why this is A\*-shaped.** E1, E2, E3 and E5 are *causal, gradable* tests of claims that vision and language have only ever tested correlationally. E4 and E6 are results the *vision* community wants: the first scale ladder at fixed data/objective, and a normalisation confound that plausibly explains part of the "only large models" folklore. Every row produces a finding whether the phenomenon is present or absent.

---

## 3. Research questions

- **RQ1 — Emergence map.** In which pretrained audio and speech encoders do high-norm artifact tokens exist, at what depth, at what rate, with what norm ratio? What separates positives from negatives: parameter scale, training compute, tokenisation (2-D patch vs 1-D frame), objective (supervised / SSL / weak supervision), attention normalisation, or **pre- vs post-LayerNorm placement**?
- **RQ2 — Adjudication (the core).** Do registers track *input information content*, *sequence position*, or *weights*? Use E1–E3, E5, E7, E8, E10 to score H-A / H-B / H-C. Sub-question only audio can pose: is the relevant variable **energy**, **entropy**, or **temporal redundancy**?
- **RQ3 — Information content.** Do audio outlier tokens lose local information (time–frequency position, patch/frame reconstruction) and gain global information (clip label from a single token), as in vision?
- **RQ4 — Training-free control.** Does `FindRegisterNeurons` + test-time registers port to artifact-positive audio encoders? How many registers does a 1500-token sequence need? Does the requirement scale with null-input extent (E3)?
- **RQ5 — Does it explain Whisper's non-speech hallucination?** Are the encoder frames the decoder cross-attends to when it fabricates text the same frames that carry high-norm tokens? Does relocating them, training-free, reduce word emission on voice-free audio **without** raising deletion cost on real speech?
- **RQ6 — Second payoff.** Do artifact-positive audio encoders inherit the INT8 PTQ hostility that ViT/BERT have, and do test-time registers restore quantisability? (This *inverts* the Zenodo preprint's finding at scale.)

---

## 4. Measurement protocol — pre-registered before any model is run

This section is load-bearing. The one prior audio measurement reached a null result partly through a measurement choice, and we will not repeat it.

### 4.1 Terminology (adopted verbatim from the 2026 cross-architectural reassessment, arXiv:2603.25803)

- **Feature map** — L2 norm of output tokens **before the final LayerNorm**.
- **Attention map** — CLS→token attention, last layer, head-averaged; for CLS-less models, **mean received attention** per token.
- **High-norm / outlier token** — per-model 98th-percentile cutoff on the pooled norm distribution (not a hand-picked absolute threshold).
- **Artifact** — an anomalous pattern in either map.
- **(ours) Null-input region** — a contiguous span of input frames that is constant by construction (Whisper zero-padding), or synthetically stationary (E2 fills).

### 4.2 Where to hook

**Hook the residual stream at each block's output, before the terminal LayerNorm.** Concretely: register forward hooks on every encoder block and read the block output tensor; do *not* read `last_hidden_state`, and do not assume a framework's `hidden_states` tuple is unnormalised — verify per model family by asserting that a known pre-LN model's final entry differs from `last_hidden_state`.

Rationale: LayerNorm rescales every token toward a common norm, so a post-LN measurement mechanically compresses max/median toward 1 whether or not artifacts exist. We will report **both** conventions for AST as an explicit methodological demonstration, and expect to recover ≈1.23 post-LN (matching Imdad) and a different, correct number pre-LN.

### 4.3 The post-LN architecture confound — must be controlled, not discovered late

Several speech SSL encoders are **post-LN**: in fairseq/HF terms `layer_norm_first=False` (wav2vec2-Base, HuBERT-Base, WavLM-Base+), versus **pre-LN** for the Large variants (`layer_norm_first=True`) and for AST and Whisper. In a post-LN stack the residual stream is renormalised at every block, so high-norm outliers have no mechanism to accumulate across depth. In the speech SSL family this is **partly confounded with Base-vs-Large**, which means a naive census would report "only large speech models have artifacts" when the real variable may be normalisation placement.

Mitigation: (i) record `layer_norm_first` per checkpoint and treat it as a factor in every analysis; (ii) find at least one **pre-LN Base** and one **post-LN Large** checkpoint to break the confound (candidates among the wav2vec2 `-960h` vs `-lv60` variants, Conformer encoders with configurable norm placement, and a synthetic control that re-normalises a Large model's stack at inference); (iii) cite arXiv:2608.09417 (*Why Post-Norm Transformers Collapse*).

### 4.4 Statistics recorded per (model, layer, clip, token)

1. L2 norm of the residual stream.
2. Mean received attention (incoming attention / uniform share), head-averaged; CLS→token where a CLS exists.
3. Per-head attention entropy.
4. **Massive-activation indicator**: any single feature with magnitude ≥ 10³ × the layer's median absolute activation (Sun et al. criterion) — a norm-independent second diagnostic.
5. Frame/patch log-energy, spectral flatness, and spectral entropy.
6. Cosine similarity to neighbours right after the patch/frame embedding (4-neighbourhood for 2-D patch models; 2-neighbourhood along time for frame models) — Darcet's redundancy measure.
7. Null-input mask (1 if the token's receptive field lies entirely in a padded or synthetic-stationary region).

### 4.5 Pre-registered artifact-positive criterion

A model is **artifact-positive** iff, at some layer, all three hold:
1. the pooled pre-LN norm distribution is **bimodal** (Hartigan dip test p < 0.01 **and** 2-component GMM beats 1-component by BIC),
2. the upper mode's median is **≥ 3×** the lower mode's median,
3. outlier positions attract **≥ 5×** their uniform share of received attention.

Otherwise **artifact-negative**. We report the continuous quantities (max/median, 98th-pct/median, outlier rate, dip statistic, attention ratio) for every model regardless, so borderline cases stay visible rather than being binarised away. Thresholds are fixed now and will not be tuned after seeing results; any deviation gets reported as a deviation.

### 4.6 Harness validation before any claim

1. Reproduce Jiang et al.'s max-patch-norm-vs-block curve on OpenCLIP ViT-B/16 using their public code, then reproduce it with **our** harness. Agreement is the entry ticket.
2. Reproduce Darcet's DINOv2-g bimodal norm histogram.
3. Reproduce Imdad's AST numbers under **their** convention, then report ours.
4. Unit-test the hooks against a hand-computed forward pass on a 3-layer toy transformer.

---

## 5. Model grid

Three axes crossed deliberately: **scale**, **tokenisation**, **objective** — plus normalisation placement as a controlled nuisance factor. "Verify" = checkpoint availability to confirm in week 1 before committing.

### 5.1 Primary arm — the scale regime where the phenomenon is expected

| Model | Enc. depth × width | Params (enc.) | Tokenisation | Pretraining | Norm | Status |
|---|---|---|---|---|---|---|
| **Whisper large-v3 encoder** | 32 × 1280 | ~635 M | 20 ms mel, conv ×2 → 1500 frames, **fixed 30 s** | weak supervision, multi-M hours | pre | `openai/whisper-large-v3` |
| Whisper large-v2 / medium / small / base / tiny | 32×1280 … 4×384 | 635 M → 8 M | same | **same corpus & objective** | pre | the **scale ladder** (E4) |
| Whisper large-v3-turbo | 32 × 1280 | ~635 M | same | distilled decoder | pre | decoder-side effects at fixed encoder |
| **XLS-R 1B / 2B** | 48 × 1280 | 1–2 B | 20 ms waveform frames | SSL, 436 k h, 128 langs | pre | largest public SSL speech encoders |
| **WavLM-Large** | 24 × 1024 | 317 M | 20 ms frames | SSL + denoising, 94 k h | pre | `microsoft/wavlm-large` |
| **HuBERT-Large / XL** | 24×1024 / 48×1280 | 317 M / ~1 B | 20 ms frames | SSL, 60 k h | pre | `-large-ll60k`, `-xlarge-ll60k` |
| **Dasheng 1.2 B** | large | 1.2 B | spectrogram patches | general-audio SSL | verify | largest general-audio encoder; non-speech at scale |
| **BEATs-Large / iter3+ (~300 M)** | ViT-L-class | ~300 M | 16×16 patches | SSL + acoustic tokeniser | verify | the missing patch-based-at-L-scale cell |

### 5.2 Negative-control arm — Base scale, incl. the correct reproduction of prior work

AST (`MIT/ast-finetuned-audioset-10-10-0.4593`), SSAST, PaSST, HTS-AT (hierarchical — the Swin/PVTv2 analogue from the reassessment), Audio-MAE, EAT, M2D, CED-Base, HuBERT-Base, WavLM-Base+, wav2vec2-Base, Whisper-small.

**EAT is the one Base model we keep as a *primary* candidate**: data2vec-style bootstrapped self-distillation is precisely the regime that produced DINOv2's artifacts, so it is the best chance of a Base-scale positive and the sharpest test of "objective matters more than scale."

### 5.3 Architecture-diversity arm — an axis vision does not have

**Conformer / FastConformer encoders** (Canary-1B, Parakeet-TDT-0.6B-v3, GigaAM): convolution modules interleaved with attention, relative position encodings, no global CLS. If registers are an artefact of pure-attention global mixing, convolution-augmented encoders should behave differently. Optional stretch: **audio Mamba / SSM encoders** (AuM, Audio Mamba), where Mamba-R showed vision SSMs have *more severe* artifacts than ViTs.

---

## 6. Datasets

**Analysis passes (forward-only, hooks on).**
- LibriSpeech test-clean + test-other, ~1000 utterances — clean read speech with natural leading/trailing silence; the natural home of Whisper padding.
- AudioSet-eval or FSD50K-eval, ~2000 clips — general sound events, mixtures, music.
- ESC-50, 500 clips — matches the prior AST measurement exactly, for comparability.
- VoxCeleb1 sample — speech in the wild, varied channel conditions.
- **Non-speech / room-tone set** for RQ5 — the benchmark released with arXiv:2609.32560 if usable, plus AudioSet non-speech classes, plus our own recorded and synthesised room tone.

**Synthetic diagnostic suite (E1–E3, E7) — our precision instrument, free to produce.**
Generated programmatically and released: speech-at-{start, middle, end, split} × pad-length {1, 3, 5, 10, 20, 29} s × fill ∈ {digital zero, white noise at −60/−40/−20 dBFS, pink noise, 1 kHz tone, looped 100 ms real fragment} × SNR sweeps × band-limited and time-masked variants. Fully specified by a seed and a config; ~10k clips, a few GB.

**Caveat to document.** Whisper's log-mel pipeline clamps with `max(log_spec, log_spec.max() − 8.0)` before scaling, so the value of a zero-padded region depends on the clip's global maximum. The pad is constant *within* a clip but not across clips. We will report pad-floor values and verify conclusions hold when the clip max is held fixed.

**Downstream (RQ3/RQ5/RQ6), frozen features only — never fine-tune the backbone.**

| Task | Dataset | Metric | Role |
|---|---|---|---|
| **Non-speech hallucination** | room tone + AudioSet non-speech + 2609.32560 arms | **word-emission rate on voice-free audio** *and* **deletion cost on real speech** (both, always) | **headline application** |
| ASR quality guard | LibriSpeech, FLEURS subset | WER | the edit must not damage transcription |
| Sound event detection (frame-level) | DESED / strong-labeled AudioSet subset | event-F1, PSDS | audio's "dense prediction" |
| Attention-as-localisation (zero-shot) | strong-labeled clips | IoU / mAP vs event boundaries | analogue of zero-shot segmentation; cheap, very visual |
| Environmental sound classification | ESC-50 (5-fold) | accuracy | no-regression |
| Keyword spotting | Speech Commands V2 | accuracy | no-regression |
| Sound event tagging | FSD50K | mAP | no-regression |
| Speaker ID | VoxCeleb1 (SUPERB SID) | accuracy | non-semantic attribute |
| Emotion | CREMA-D | accuracy | paralinguistic |
| **INT8 PTQ fidelity** | above, ×3 backends | token cosine to FP32, task metric, footprint | RQ6 |

---

## 7. Phase plan (gated, not calendared)

Each phase names its **gate** — what must be true to proceed. Week counts are effort estimates, not commitments.

### Phase 0 — Harness + the emergence gate (~1–1.5 weeks)
Build the hook library; pass all four validations in §4.6; verify checkpoint availability across §5. Then run the **gate experiment**: pre-LN norm statistics over every block for the primary arm, on 500 clips including the synthetic pad sweep.

**Gate (pre-registered decision rule):**
- **Green** — any primary model reaches max/median ≥ 5× with a bimodal distribution and outliers concentrated on null-input or low-energy frames → proceed to the full programme; the paper is C1–C4 (§9) with RQ5 as headline. Whisper large-v3 is where we expect this.
- **Amber** — ratios in the 2–5× band, or positive only in the very largest models → the paper becomes the **emergence-map + adjudication** paper (RQ1, RQ2, RQ6); RQ5 demotes to a section. Interspeech / A\* workshop, ACM MM as stretch.
- **Red** — everything under 2×, including Whisper large-v3, XLS-R-2B and Dasheng → **do not run the v1 programme.** Pivot immediately to the negative-result mechanism paper: E2, E3, E6, E10 plus the emergence-condition analysis (softmax normalisation, pre/post-LN, bidirectional vs causal, training compute), which is a genuine and publishable statement about *when* the phenomenon does and does not arise. Target A\* workshop or Interspeech.

This is the single highest-value ~2 days in the project. It runs **before** probes, benchmarks, or any training.

### Phase 1 — Emergence map (~2.5–3 weeks)
Full grid × all analysis datasets. Factorial analysis of artifact-positivity against scale, compute, tokenisation, objective, architecture class and **norm placement**. Break the post-LN/Base confound (§4.3). Deliverable: the emergence-map table + figure gallery (spectrogram / norm-map / attention-map triptychs) + the corrected AST reproduction.

**Gate:** at least one clean contrast in hand — a positive/negative pair differing in one factor.

### Phase 2 — The dissociation battery (~3.5–4 weeks) — *the core of the paper*
E1, E2, E3, E5, E7, E8. Score H-A / H-B / H-C against the §2 matrix. Report an explicit adjudication with effect sizes and confidence intervals, and state honestly which rows are inconclusive.

**Gate:** the matrix resolves at least the content-vs-position axis (E1 is the decisive row and is cheap).

### Phase 3 — Mechanism (~2 weeks)
E9: port `FindRegisterNeurons` (top-k over MLP hidden units at outlier positions; ablate k ∈ {5, 10, 20}; search layers up to the emergence layer). Validate causality by *moving* an outlier to a chosen frame; run both negative controls (random-neuron intervention should do nothing; zero-without-move should hurt, quantified on ASR/ESC-50). E10: singular-defect test, including an input-independence measurement of the high-norm direction. For GLU-MLP models (WavLM, BEATs variants), replicate the Llama-AVSR GLU-gating origin analysis.

**Gate:** a causal neuron set exists in ≥1 model, or we can show it does not and say why.

### Phase 4 — Information probes (~1.5–2 weeks)
RQ3 with the reassessment's exact hyperparameters (linear heads; CE + 0.5·MSE for position; MSE for reconstruction; AdamW, cosine schedule, ≤30 epochs, early stop patience 3): (a) predict a token's (time, frequency) grid position or time-index bucket; (b) reconstruct its log-mel patch / frame slice; (c) classify the clip from a **single** token (CLS vs random-normal vs random-outlier vs test-time register). Feature extraction dominates cost; cache once.

### Phase 5 — Applications (~3.5–4 weeks)
**5a — Whisper non-speech hallucination (headline).** Establish the mechanistic link: do the encoder frames the decoder cross-attends to during fabrication coincide with high-norm frames? Then intervene: append test-time register(s), relocate register-neuron activations out of the null-input frames, re-decode. Report **word-emission rate on voice-free audio and deletion cost on real speech together** — the 2026 literature is explicit that suppression-only numbers are misleading. Baselines: unmodified Whisper; VAD preprocessing; null-token logit shift (arXiv:2608.15940); hallucination-space projection (arXiv:2609.04561); the fine-tuned checkpoints of arXiv:2609.32560. Our selling point is *training-free*, so we must win on the cost axis even where we lose on raw suppression.
**5b — Frame-level SED + attention-as-localisation**, pre/post edit.
**5c — INT8 PTQ (RQ6)** across x86/ARM/CUDA backends, pre/post edit; direct engagement with the Zenodo preprint's diagnostic proposal and with arXiv:2510.04547 (registers for vision-encoder activation quantisation).
**5d — No-regression suite** on ESC-50 / SC-V2 / FSD50K / VoxCeleb1 / CREMA-D. Null results reported as results ("the edit is safe on N tasks").

### Phase 6 — Optional, the only training phase (~20–40 A100-h)
PH-Reg-style self-distilled registers for the 1–2 most artifact-positive encoders. Teacher denoising by **time-shift TTA only** — frequency shift is not label-preserving in audio, and documenting that asymmetry is itself a contribution. Student = frozen weights + m registers, unfreezing registers + positional embeddings; cosine + MSE on dense features over unlabeled AudioSet. Gives the "training-free vs cheap-training" comparison reviewers ask for. **Skip and state as future work if Phase 5 is already strong.**

### Phase 7 — Consolidation
Statistics, figures, the released toolkit, writing, internal review, submission.

---

## 8. Related work: the differentiation map

Every line here is verified as of 2026-09-30. Each entry carries the sentence we will use to distinguish ourselves.

### 8.1 The one prior audio measurement — handle carefully and generously
**Imdad, *Silence in the Noise*** (Zenodo, 2026-06-16, DOI `10.5281/zenodo.20710635`). One checkpoint (AST), one dataset (ESC-50), 64 clips; trained Darcet-style registers fine-tuned on ESC-50 (+0.33 pp, p = 0.12); INT8 PTQ across three backends; proposes max/median final-layer norm ratio as an INT8-readiness diagnostic. Reports AST outlier-free.

*Our line:* "The only prior measurement of this phenomenon in audio evaluates a single Base-scale checkpoint on one dataset and computes token norms **after** the terminal LayerNorm, a convention under which the max-to-median ratio is compressed toward unity by construction. We reproduce their number under their convention, report the corrected pre-LayerNorm measurement, and extend the analysis to N checkpoints spanning 20 M–2 B parameters, both tokenisations, and the frame-based speech encoders their study does not consider."

Also: their limitations section explicitly names dense audio prediction as the right setting and expects larger effects there — i.e. our Phase 5b is pre-endorsed by the closest prior work. Their INT8 finding, if it inverts at scale (RQ6), becomes a result rather than a conflict.

### 8.2 Closest audio-side work (cite and differentiate)
- **arXiv:2504.01690 / IEEE OJSP 2025, Lee et al., *Token Pruning in Audio Transformers*** — **the closest prior work to RQ2's correlational half.** Reports high correlation between AST/AudioMAE attention scores and patch intensity/variation; finds low-intensity tokens remain important; AudioMAE retains more low-intensity tokens than AST (attributed to the reconstruction objective). *Our line:* they measure attention against intensity, correlationally, for a pruning objective; we measure **norms**, add causal interventions on constructed null input, and separate energy from information from redundancy.
- **arXiv:2507.02666 ASDA** — differential attention for audio SSL, motivated by attention "allocated to irrelevant information"; SOTA on AS-2M / SPC-2 / ESC-50; no norm analysis. *Our line:* a competing **architectural** fix for what may be the same pathology, requiring pretraining; we diagnose the phenomenon and fix it training-free in existing checkpoints.
- **arXiv:2406.11022** Whisper PTQ with gated attention — confirms outliers in Whisper **weights and activation tensors**. *Evidence in our favour*; differentiate on token-level vs tensor-level and on characterisation vs compression.
- **arXiv:2504.14915 StableQuant** — layer-adaptive PTQ for HuBERT/wav2vec2 driven by per-layer scale distributions. Same category of indirect evidence.
- **arXiv:2510.22603** Llama-AVSR sinks — LLM **decoder**, sinks at BOS and low-semantic prompt tokens, MLP-GLU gating origin, fine-tuning-based fix. Borrow the GLU analysis and the 10³×-median criterion; differentiate on encoder-vs-decoder and training-free-vs-loss-term.
- **arXiv:2602.23702 / 2606.21268** Online Registers for streaming S3Ms — registers as architectural placeholders for missing future context; no artifact analysis. Forces the wording: registers have appeared in speech **as a design device**; nobody has asked whether pretrained audio encoders develop the artifacts registers were invented to fix.
- **arXiv:2609.20849 SPARE** — one register token in a Large Audio Language Model, supervised by a Sentence-BERT cosine loss. Register-as-device again.
- **arXiv:2605.10815 / 2603.14337** — sink tokens as cross-modal hubs in AV-LLMs and Omni-LLMs. Decoder-side.
- **arXiv:2205.03759 *Silence is Sweeter Than Speech*** — HuBERT stores **speaker** information at silence positions. Now load-bearing: direct prior evidence that a speech encoder repurposes silence frames for global storage. Promote from motivation to a cited prediction we test properly.
- **arXiv:2407.04439 XLSR-Transducer** — exploits sinks to halve left context in streaming ASR. A tool, not an analysis.
- **arXiv:2607.00387** *From Objectives to Applications: Aligning Architectural Biases in Audio SSL* — framing citation for our objective × architecture contrasts.

### 8.3 The Whisper-hallucination baselines (RQ5 must be positioned against these)
- **arXiv:2609.32560 *How to Reduce Whisper Hallucination*** (2026-09-26) — 61.9% word emission on 42 pure-room-tone clips, 100% emit something; 11,852-clip benchmark across 8 arms; fixes by synthetic-positive TTS augmentation + fine-tuning; emphasises that suppression-only evaluation hides a 58% deletion rate on repeated speech.
- **arXiv:2609.04561** hallucination-space projection — decoder states, decode-time.
- **arXiv:2608.15940 *The Null Token Knows*** — decoder null-token logit shifts; explicitly does **not** cite Darcet and does **not** analyse norms, sinks or registers.
- **arXiv:2607.01108 NPUsper** — detects hallucinated tokens from decoder temporal patterns.

*Our line, and the reason this is a real contribution:* **every published remedy operates on the decoder or the training data. None asks what the encoder did with the silence.**

### 8.4 Vision / language side (the niche has moved a lot since July 2026)
- Anchors: **2309.16588** Darcet (ICLR 2024), **2506.08010** Jiang (NeurIPS 2025 spotlight; the method we port; public code), **2505.21501** PH-Reg, **2401.02957** DVT, **2603.25803** cross-architectural reassessment (our terminology and probe hyperparameters).
- **2602.22394 *Vision Transformers Need More Than Registers*** — artifacts as **lazy aggregation**: ViTs use background patches as shortcuts for global semantics; fixed by selective patch→CLS integration; 12 benchmarks. A refined form of H-A and a direct competitor for the mechanism story. Engage head-on.
- **2605.19622 UniRefiner** — argues the "high-norm token" definition is too narrow.
- **2607.16824** Test-Time Registers as Global Priors for Tokenized Image Generation — the method we port, ported elsewhere first.
- **2604.14433** Zero-ablation overstates register content dependence — methodological warning for our ablations: prefer *moving* to *zeroing*.
- **2605.16147 / 2605.05206** — pixel-space DiTs and DiT outliers: the phenomenon is **absent** in DiTs. Supports "heterogeneity is a finding."
- **2405.14858 Mamba-R** — vision SSMs have *more severe* artifacts. Motivates the audio-Mamba stretch arm.
- **2502.07004 *Demystifying Singular Defects in LLMs*** (ICML 2025) — high-norm tokens via singular vectors of layer linear approximations. **The operational form of H-C, and the source of E10.**
- **2603.17771 *Attention Sinks Induce Gradient Sinks*** — massive activations as RMSNorm-mediated gradient regulators, **under causal masking**. Audio encoders are bidirectional, so this predicts encoder-vs-decoder asymmetry → **E5**.
- **2410.10781** When Attention Sink Emerges (ICLR 2025 spotlight), **2402.17762** Massive Activations (COLM 2024), **2309.17453** StreamingLLM, Barbero et al. *Why do LLMs attend to the first token?* (COLM 2025) — H-B.
- **2608.09417** *Why Post-Norm Transformers Collapse* — §4.3.
- **2604.10098** attention-sink survey — verified to contain **no audio-encoder category**. Cite as evidence of the gap.
- **2510.04547** Activation Quantization of Vision Encoders Needs Prefixing Registers — the vision precedent for RQ6.
- **2507.16018**, **2501.04784**, **2505.05892**, **2509.20986 SiNGER**, **2608.10989**, **2606.14701 RATS**, **2606.12036** — the wider register ecosystem; skim for framing and cite where relevant.

---

## 9. Contributions, as they will appear in the paper

- **C1 — Emergence map.** The first multi-family, multi-tokenisation, **multi-scale** census of high-norm artifact tokens in pretrained audio and speech encoders (18–22 checkpoints, 20 M–2 B, patch vs frame, supervised / SSL / weak supervision, pure-attention vs Conformer), under a corrected pre-LayerNorm protocol, with the post-LN confound controlled — plus the corrected reproduction of the only prior audio measurement.
- **C2 — Adjudication.** The first *causal, gradable* dissociation of content, position and weights as determinants of register placement, using constructed null input in Whisper's fixed 30-second window, the energy × information × redundancy fill battery, the pad-length sweep, and the bidirectional-encoder / causal-decoder contrast within a single checkpoint. **These experiments cannot be run in vision or language.**
- **C3 — Application: an encoder-side account of Whisper non-speech hallucination**, plus a **training-free** intervention benchmarked against the 2026 decoder-side and data-side baselines on both suppression *and* deletion cost.
- **C4 — Second application and a no-regression guarantee.** Frame-level SED and attention-as-localisation gains; INT8 PTQ quantisability restored by test-time registers; no regression across five clip-level tasks.
- **C5 — Released toolkit.** `audio-registers`: hooks, diagnostics, the pre-registered criterion, the synthetic diagnostic generator, and per-checkpoint emergence reports — reusable for any new audio encoder, which is how this gets cited.

### What we will *not* claim
- Not "registers have never appeared in speech models" (they have, as a design device).
- Not "nobody has measured artifacts in an audio transformer" (Imdad has, for AST).
- Not causality for hallucination from correlation alone — the claim requires the Phase 5a intervention to move the metric.
- Not that our fix beats fine-tuned baselines on raw suppression; the claim is a better cost/benefit at zero training.

---

## 10. Titles under consideration

1. **Where Does a Transformer Put the Silence? Dissociating Content, Position, and Scale in Register-Token Emergence** *(preferred — leads with the mechanism, signals the instrument)*
2. **Registers in the Quiet: Audio Encoders as a Testbed for Why Transformers Grow High-Norm Tokens**
3. **Do Audio Transformers Need Registers? An Emergence Map and an Encoder-Side Account of Silence Hallucination** *(keeps the v1 hook; better for Interspeech/ICASSP than for ICML)*
4. **The Padding Is the Register: High-Norm Tokens in Non-Speech Frames of Speech Encoders** *(strongest if E1 lands cleanly)*

---

## 11. Venues, ordered by fit

No deadline pressure; the ladder is about *fit*, and we submit when the mechanism story is complete.

| Venue | Tier | Cadence | Fit |
|---|---|---|---|
| **ICML** | A\* | annual, ~Jan | Best fit for C2 as framed — a mechanism claim about transformers, audio as apparatus. |
| **NeurIPS** | A\* | annual, ~May | Equally good; more room for the full battery + Phase 6. |
| **ICLR** | A\* | annual, ~Sept | Strong fit — Darcet and Jiang both landed in this lineage. |
| **ACM Multimedia** | A\* | annual, ~April | Takes audio seriously; C3/C4 land well if the application half dominates. |
| **ICLR / ICML / NeurIPS workshops** | A\* workshop | 2–3 windows/yr | **The floor.** Every branch of the gate, including Red, produces a workshop paper. |
| **Interspeech** | A | annual, ~Feb | Guaranteed landing zone; ideal for the emergence map + hallucination result if the mechanism story stays thin. |
| **ICASSP** | B | annual, ~Sept | 4-page version of C1 + C3. |
| **TASLP / TMLR** | Q1 journal / no-deadline | rolling | Extended version. TMLR is genuinely attractive given no deadline pressure and a reproducibility-heavy contribution. |

Strategy: build toward the strongest A\* submission, and **arXiv the emergence map early** once Phase 1 is done, to timestamp priority without waiting for the full programme.

---

## 12. Compute and storage

| Phase | Nature | Estimate |
|---|---|---|
| 0 gate | inference w/ hooks, primary arm, 500 clips | 3–6 T4-h |
| 1 emergence map | inference, ~20 models × ~4k clips | 25–40 T4-h |
| 2 dissociation battery | inference, synthetic suite | 15–25 T4-h |
| 3 mechanism | neuron search, interventions, SVD of layer maps | 10–20 T4-h |
| 4 probes | one extraction pass + linear heads | 10–15 T4-h |
| 5 applications | decode + feature extraction ×2 (pre/post), 3 quant backends | 25–40 T4-h |
| 6 PH-Reg (optional) | distillation, registers + pos-emb only | **20–40 A100-h** |
| **Total** | | **~90–150 free Kaggle GPU-hours + optional small A100 budget** |

Kaggle free tier is 30 GPU-h/week, so compute is not the bottleneck; **session limits and feature-cache storage are.** Shard extraction per (model, dataset); checkpoint features to Kaggle datasets between 12-hour sessions; prune per-layer caches to the layers actually needed once Phase 1 identifies emergence depths. Budget ~150–300 GB. All models fit fp16 inference in <8 GB, including the XLS-R-2B encoder. Local RTX 4050 (6 GB) is for development and small-model debugging only.

---

## 13. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Nothing is artifact-positive anywhere, including at 2 B scale** | moderate — the single biggest risk, and it is scientific, not competitive | Phase 0 gate detects it in ~2 days for a few GPU-hours. Red branch is a pre-planned pivot to the negative-result mechanism paper, not a salvage operation. |
| Scooped by an ICLR 2027 submission (deadline passed 2026-09-25, listing not yet public) | low–moderate | Re-check OpenReview when the listing opens; re-run the §1 sweeps monthly; arXiv the emergence map after Phase 1. Our moat is the dissociation battery, which needs a synthetic suite nobody else has built. |
| Structured search misses another Zenodo/OSF-class preprint | **moderate — it already happened once** | Every sweep now pairs S2 + arXiv with a plain web search on the exact phrasings, plus Scholar alerts on "register tokens audio", "attention artifacts speech", "high-norm tokens audio". |
| Post-LN confound makes the census uninterpretable | **certain unless controlled** | §4.3: record `layer_norm_first` per checkpoint, treat as a factor, break the confound with pre-LN-Base / post-LN-Large pairs and a synthetic re-normalisation control. |
| Whisper mel clamping makes "zero pad" clip-dependent | certain | Documented in §6; verify conclusions with clip max held fixed. |
| Frame models have no CLS, 1-D, variable length | certain | Reassessment's mean-received-attention workaround; treat length as a variable to exploit (E8), not a nuisance. |
| Hallucination intervention improves suppression but raises deletion | plausible | Report both axes always. The 2026 literature already established that suppression-only numbers mislead, so this is a known-shape evaluation and an honest cost curve is itself a contribution. |
| "This is just applying a vision result to audio" | **certain to be raised** | The abstract leads with C2, not C1. The dissociation experiments are impossible in vision; say so in the first paragraph and put the prediction matrix in Figure 1. |
| Kaggle 12-h session limits vs long extraction | certain | Shard and checkpoint; treat the feature cache as a first-class artifact. |
| Phase 6 A100 budget not approved | plausible | Explicitly optional; state as future work if Phase 5 is strong. |

---

## 14. What to ask the supervisor

1. **Settle the venue ambition explicitly.** v1 targeted ICASSP (CORE B) and Interspeech (A). If A\* is the goal, the paper must be a mechanism paper (C2-led), not a modality census — that is a different paper, and this plan is the different paper. Confirm which one we are writing.
2. Green light on the **gated** structure, specifically the authority to execute the Red pivot on Phase 0 evidence without renegotiating scope.
3. ~40 A100-hours for Phase 6 (optional) and ~300 GB storage.
4. A read on whether to **arXiv the Phase 1 emergence map early** for priority, given an active vision-side niche and an ICLR 2027 listing we cannot yet see.
5. Weekly 30-minute check-ins; the **Phase 0 gate result** is the first formal go/no-go, and the Phase 2 adjudication matrix is the second.

---

## 15. Reading list (revised, in order)

1. **Darcet et al., ICLR 2024** — the phenomenon. Read fully. `2309.16588`
2. **Jiang et al., NeurIPS 2025 spotlight** — the method we port; clone the code, run the demo, reproduce their Figure 1 before writing any of our own hooks. `2506.08010`
3. **Cross-architectural reassessment, 2026** — §3 terminology and probe hyperparameters, adopted verbatim. `2603.25803`
4. **Imdad, *Silence in the Noise*** — the prior audio measurement; read the Method section closely for the LayerNorm convention. Zenodo `10.5281/zenodo.20710635`
5. **Whisper** — §2 architecture, and specifically the feature extractor's 30-second padding and log-mel clamping. This is the apparatus. `2212.04356`
6. **Wang et al., *Demystifying Singular Defects in LLMs*** (ICML 2025) — H-C and E10. `2502.07004`
7. **Attention Sinks Induce Gradient Sinks** (2026) — the causal-masking prediction behind E5. `2603.17771`
8. **Gu et al., *When Attention Sink Emerges*** (ICLR 2025) + **Sun et al., *Massive Activations*** (COLM 2024) + **StreamingLLM** — H-B. `2410.10781`, `2402.17762`, `2309.17453`
9. **Vision Transformers Need More Than Registers** — the strongest competing form of H-A. `2602.22394`
10. **Lee et al., *Token Pruning in Audio Transformers*** — closest to RQ2's correlational half; know exactly what they did. `2504.01690`
11. **How to Reduce Whisper Hallucination** + **The Null Token Knows** + **Hallucination Space Projection** — the RQ5 baselines and their evaluation protocol. `2609.32560`, `2608.15940`, `2609.04561`
12. **AST** (`2104.01778`) — bridge model, §2 carefully. **PH-Reg** (`2505.21501`) — method section only, for Phase 6.
13. Architecture/tokenisation skims: HuBERT `2106.07447`, WavLM `2110.13900`, XLS-R `2111.09296`, BEATs `2212.09058`, Audio-MAE `2207.06405`, EAT `2401.03497`, Dasheng, Conformer/FastConformer.
14. **Silence is Sweeter Than Speech** `2205.03759` — prior evidence that HuBERT stores speaker identity in silence.

*Audio prerequisites remain minimal:* one afternoon with `librosa.display.specshow` until silence, harmonic stacks and broadband noise are visually obvious. No Fourier theory, filter design or vocoders required — the frontends are frozen and all the analysis lives in transformer-internals space.

---

## 16. Reproducibility commitments

- Thresholds in §4.5 and the gate rule in Phase 0 are **fixed before data collection** and restated verbatim in the paper; any deviation is reported as a deviation.
- Every number traced to a script and a seed; 3 seeds per probe, mean ± std, 95% CIs; paired bootstrap over clips for all before/after deltas.
- Synthetic diagnostic suite fully specified by a config + seed and released as a generator, not as a blob.
- `audio-registers` toolkit released with per-checkpoint emergence reports, so the census is extensible rather than a snapshot.
- Negative results reported at equal prominence — including the Base-scale negative-control arm and any inconclusive row of the §2 matrix.

---

*Prior-art appendix, verification queries, citation-graph counts and the full differentiation notes live in `GapVerification_2026-09-30.md`. Anchor-paper TeX sources are in `Papers_Tex_Source/`, PDFs in `Papers_pdf/`. Re-run the sweeps before submission; structured indices did not find the one existing audio paper, so structured indices will not find the next one either.*
