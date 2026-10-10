# Evidence log — verified facts behind `plan.md`

Every entry below was written from the **full TeX source** (or PDF where arXiv only has a template) of the paper,
downloaded with `scripts/download_arxiv_papers.py`, or from our own pilot runs. Section/table references point into
the original paper. Nothing here is from abstracts alone unless explicitly marked. IDs are arXiv IDs; venue status
was checked on OpenReview / Papers-with-Code / arXiv comments on 2026-10-10.

**How to use:** `plan.md` cites entries as `[E07]`. When you add evidence, append a new `E##` entry with the same format:
*ID · authors · title · venue — facts (with section refs) — implication for us.* Mark anything unverified as **UNVERIFIED**.

## Synthesis (what the evidence supports, 2026-10-11)

1. **The sink / massive-activation / register literature is the hottest mechanism topic at A* venues in 2026**
   (ICML 2026: ≥24 accepted papers; ICLR 2027: 74 submissions) — but **no paper studies these phenomena inside speech/audio encoders**,
   silence, or Whisper's 30-s padding ([scoop check](scoop_check_2026-10-10.md)). [E03, E04, E06, E18, E19]
2. **2026 theory makes conflicting predictions that speech can separate**: sinks ≠ massive activations (pre-norm makes them co-occur) [E03];
   sinks arise from the causal first token's unaggregated variance [E04]; a "default/no-op" state forces a sink [E13];
   spikes go to the least-shared/smallest token [E06]; artifacts are lazy aggregation from coarse supervision [E16];
   outliers emerge when meaningless tokens become identifiable [E02]. Speech has *exactly constant* padding frames, digital silence,
   stationary noise and tones — controllable redundancy/energy/position that images and text lack.
3. **Our pilots (Whisper base/small/large-v3-turbo, n=100 each) show** [E22, E23]: (i) Whisper's encoder forms massive-activation
   "register" frames on padding (Ψ = 1,122 in base); (ii) these high-norm frames are **dispensable** for ASR; (iii) what Whisper needs
   from padding is **silence used two ways** — the decoder needs silent encoder states as end-of-speech evidence, and the encoder's
   speech frames attend to a few **position-anchored sink frames** at the start/end of the 30-s window; (iv) a **training-free cached-silence**
   scheme (precomputed silence K/V + tail) matches full-padding WER at ≈0.16× encoder FLOPs and fixes runaway repetition (nopad WER 214–428%).
4. **Practical prior art for padding-free Whisper all needs training** (hush word [E08], WhisperKit block masks + self-distillation [E09],
   FUTO ACFT fine-tuning, ICASSP'26 FT+KD) or is orthogonal (NPUsper online stopping [E10], stride-k token dropping [E12]).
5. **Audio LLMs inherit the Whisper encoder in three padding regimes** (visible: Whisper; masked: Qwen2-Audio, Audio Flamingo 3; none: Qwen2.5-Omni windows) [E12] —
   a natural experiment no one has run.

## Index
| ID | Paper (arXiv) | Venue / status (checked 2026-10-10) | Role in our plan |
|---|---|---|---|
| E01 | Jiang et al., *ViTs Don't Need Trained Registers* (2506.08010) | NeurIPS 2025 spotlight | test-time register method; register neurons; attention-bias variant |
| E02 | Kim et al., *RegCache* (2510.04547) | ECCV 2026 | cached-KV registers for PTQ; background-zeroing → earlier outliers |
| E03 | Sun, Canziani, LeCun, Zhu, *The Spike, the Sparse and the Sink* (2603.05498) | ICML 2026 | sinks ≠ massive activations; pre-norm coupling; self-sinking |
| E04 | *Structural Origin of Attention Sink* (2605.06611) | ICML 2026 | causal-mask variance theory; mask-to-self intervention |
| E05 | Luo et al., *To Sink or Not to Sink* (2510.08510) | ICLR 2026 | encoder sinks propagate into the LLM (vision) |
| E06 | Ngnawé et al., *Mind the Spike* (2609.32808) | arXiv 2026-09; ICLR 2027 submission | criteria (Ψ vs φ); least-shared location rule; trigger removal |
| E07 | *Audio Token Attention Is Predictable* / Triage (2609.38878) | arXiv 2026-09; ICLR 2027 submission | LALM attention; encoder norm anti-correlated with LLM attention |
| E08 | Wang, Xu, Lin, *WhisperFlow* (2412.11272) | systems paper (PDF) | padding-length table; trained "hush word" baseline |
| E09 | Argmax, *WhisperKit* (2507.10860) | ICML 2025 (per authors) | silence caching with block masks + self-distillation (trained) |
| E10 | *NPUsper* (2607.01108) | arXiv 2026-07 | unpadded failure = decoder termination; cross-attn hallucination stop |
| E11 | Darcet et al., *ViTs Need Registers* (2309.16588) | ICLR 2024 (origin only) | phenomenon origin; redundancy hypothesis |
| E12 | *Stride-k Subsampling for Whisper* (2608.30927) | EMNLP 2026 | Whisper 1500-token interface; LALM padding regimes (with HF code check) |
| E13 | *Attention Sinks Are Provably Necessary* (2603.11487) | ACL 2026 | default-state / no-op sink theory |
| E14 | Sun et al., *Massive Activations in LLMs* (2402.17762) | COLM 2024 (definition only) | massive-activation criterion |
| E15 | Radford et al., *Whisper* (2212.04356) | ICML 2023 (facts) | architecture, padding, head dim 64 |
| E16 | *ViTs Need More Than Registers* / LaSt-ViT (2602.22394) | CVPR 2026 | lazy aggregation; Patch Score / Point-in-Box |
| E17 | *Demystifying Singular Defects in LLMs* (2502.07004) | ICML 2025 | BERT has no high-norm tokens; causal-attention conjecture (caveat) |
| E18 | *Attention Sinks and Compression Valleys* (2510.06477) | ICLR 2026 | entropy valley ↔ massive activations |
| E19 | *A Single Layer to Explain Them All* (2605.08504) | ICML 2026 | massive-emergence (ME) layer |
| E20 | Anand et al., *Sinks in AVSR LLMs* (2510.22603) | ICASSP 2026 | closest audio work (LLM side only) |
| E21 | *Cross-modal Information Hubs in AV-LLMs* (2605.10815) | ICML 2026 | closest audio work (LLM side only); scoop watch |
| E22 | **Our pilot v1** — padding conditions, register transplant/masking | this repo | registers neither necessary nor sufficient (Whisper-base) |
| E23 | **Our pilot v2** — cached-silence interventions, 3 Whisper sizes | this repo | silence = decoder evidence + position-anchored encoder sinks |

---

## E01 [2506.08010] Jiang, Dravid, Efros, Gandelsman — "Vision Transformers Don't Need *Trained* Registers" — NeurIPS 2025 (spotlight)
- Mechanism: a sparse set of MLP "register neurons" (post-nonlinearity hidden units) in the layers *before* outliers appear creates the high-norm tokens; outliers appear right after one MLP (OpenCLIP-B/16: layer 6; DINOv2-L: layer 17) and CLS-attention sinks appear in the next layers (§3.1, App. DINOv2).
- FindRegisterNeurons (Alg.1): average activation of each neuron at outlier positions (norm > threshold) over a dataset, layers ≤ top_layer, return top_k. OpenCLIP-B/16: threshold 75, top_layer 5, top_k 10. DINOv2-L/14: threshold 150, highest layer 17, top_k 45. Heuristic: threshold = mean + 3σ of patch norms; top_layer = layer where outliers first appear; sweep top_k until outliers suppressed (App. hyperparameters).
- ShiftRegisterNeurons (Alg.2): append one zero-initialised token; for each register neuron copy the max activation over tokens into that token and zero the neuron elsewhere. Initialisation (zero/random/mean) does not matter (App.).
- Causal: zeroing the 10 register neurons drops OpenCLIP-B/16 zero-shot IN1k 70.4→55.6 (random 10 neurons: 69.3±1.1) — the outlier must live *somewhere* (§4, §6.2).
- Test-time register ≈ trained register: DINOv2-L linear probe IN 86.4 (=base), ADE20k 48.3→49.1, NYUd 0.388→0.378; LOST corloc VOC07 32.2→53.8 (trained reg 56.2). OpenCLIP gains marginal (LOST 30.8→30.9).
- One register suffices (cos-sim of value update 0.985 at 5 regs; NYUd best at 1–2).
- Attention-bias variant (App.): set k′,v′ = mean key/value of the test-time register over 1000 images, zero register neurons → keeps 71.3% zero-shot and removes outliers from the residual stream in all layers. Supports "registers function mainly as attention biases" (Sun et al.).
- VLM: LLaVA-Llama-3-8B, 100 register neurons out of ~100K in the CLIP-L encoder; benchmark avg 46.2→46.2; the register is NOT passed to the LLM ("we leave exploring its use as global memory for the LLM … to future work"); outliers "leak into the language model's attention" (qualitative).
- InternViT-6B: outlier norms up to 7000 vs median 332; 300/576k register neurons; max norm → 608; LOST +11–13 corloc.
- Limitation they state: only neurons studied; positional encodings may also contribute.
- **Implication:** the method is fully training-free and inference-only → runs on a T4/RTX-3050 for Whisper-tiny…large-v3. Nobody has applied it to audio (verified by arXiv/S2/ICLR-27 scan). Their "future work" (register → LLM) is exactly the LALM interface question for Whisper-encoder LALMs.

## E02 [2510.04547] Kim, Yeom, Kim, Park, Kim, Lee (POSTECH/Google) — "Activation Quantization of Vision Encoders Needs Prefixing Registers" (RegCache) — ECCV 2026 (arXiv comment; template says CVPR 2026 submission)
- Observation boxed in §1: sink tokens emerge gradually from the MIDDLE layers of vision encoders, giving rise to outliers; these tokens are highly similar across images (cos 0.89±0.07 vs normal tokens 0.26±0.10, SigLIP-B/16, Tab. cos-sim) → reusable registers.
- Quantization-sensitive layers (W8A8 one layer at a time) are the MLP (FC2) layers in 1–2 middle blocks = where max ℓ∞ of FC2 inputs jumps (§3.1, Fig. sensitivity).
- **Causal background test (§3.2, Fig. FG/BG):** zeroing background pixels (foreground-only images) makes outliers emerge EARLIER and with LARGER magnitude; background-only ≈ original. Interpretation: outliers emerge once "semantically meaningless" tokens become identifiable; in LLMs ⟨BOS⟩ is meaningless from the input → early-layer outliers.
- App. "tokenization perspective": DINOv2 *with* 4 trained registers shows outliers from layer 2 (LLM-like), without registers from middle layers → "closed-set" meaningless tokens move outlier emergence early.
- Method: (1) curate top-k=100 ℓ∞ tokens at the sensitive layer from 50k ImageNet train images (+3 preceding blocks); (2) average their KV caches and insert τ∈{1..15} copies as prefix KV from a few layers before the sensitive layer (DINOv2: only at that layer); (3) delete top-k̃ ℓ∞ tokens at the sensitive block input (≤10). Search ≈1 h on RTX 4090.
- Results: e.g. W8A8 zero-shot IN1k CLIP-B/16 naive 34.01 → 59.71; SigLIP2 26.04 → 72.35; W4A4 with RepQ-ViT CLIP 1.83 → 21.50; Qwen3-VL-2B 4-bit VQAv2 32.03 → 42.72. Latency overhead ≤1.54%. Max token norm at sensitive layer: OpenCLIP 92.78 → 9.64.
- Ablation: prefix-only or delete-only can be WORSE than naive (SigLIP2: 23.82 / 69.06 vs 72.35 combined).
- vs test-time register (Jiang KV variant) under W8A8: CLIP-B/16 naive 34.01, TTR 44.30, RegCache 59.71 (App.).
- Simpler baselines: random sink prefix 54.18, outlier clipping 47.29 (CLIP W8A8).
- Limitations: many hyperparameters searched per model/bit-width.
- **Implication:** (a) a ready PTQ method/baseline to port to Whisper; Interspeech-2026 quantization papers (DiffAQ, ESC) report "zero-padded regions dominate calibration" and "large audio calibration ranges" without a register explanation. (b) Strong testable prediction for audio: Whisper's padding frames are *exactly constant* input vectors (log-mel clamp) → "meaningless tokens identifiable from the input" → registers should appear EARLY (LLM-like), unlike ViTs; silence-free speech should push emergence later.

## E03 [2603.05498] Sun, Canziani, LeCun, Zhu (NYU) — "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks" — ICML 2026 (pwc conference tag; OpenReview ICML'26 accepted list)
- Five properties of massive activations (§3.1): intermediate layers only; few channels; channels spike together; fixed inter-channel ratios; few tokens.
- Life cycle "rise–plateau–fall": one/two early **step-up blocks** inject them, residual stream carries them, late **step-down blocks** cancel with opposite sign (Tab. step-up/down: e.g., Llama-2-7B step-up block 4, step-down 62 of 64).
- FFN (SwiGLU) = "directional quadratic amplifier": per output channel k a quadratic form U_k; spike channels have one dominant eigenvalue with a shared eigenvector s* → any token aligned with s* spikes in all spike channels.
- Why first tokens spike (Tab. ubiquity: 98.4–99.9% of the vocabulary spikes at position 0): the first token attends only to itself, so attention is a **static linear map W_VO** that steers it toward s*. Delimiters do the same via **self-sinking** (attend to themselves). "A token transitions into a spike token when it demonstrates a strong self-sinking bias in early layers."
- Pre-norm RMSNorm maps spike tokens to sparse, near-constant vectors → keys collapse to a 1–2-D subspace → sink heads.
- Causal ablations (7B Llama-style, 100B tokens, from scratch): sandwich norm / QK-norm / DynamicTanh suppress spikes (3818 → 520 / 92 / 153) but sinks persist (sink ratio 44.7/42.0/61.0% vs 46.0%) → **sinks do not require massive activations**; head dimension is the main driver of sinks (d_head 8 → 4.1% sinks; 128 → 46.0%); conditional per-head/per-channel gating removes sinks (6.4%/4.5%); training only on long contexts collapses sinks (1.2–13%).
- Sink ratio definition follows Gu et al. 2024 (importance score α_k = mean attention received; threshold).
- **Implication:** (i) the pre-LN/post-LN split predicts our pilot: post-LN SSL-Base encoders should show sinks without norm spikes; pre-LN Whisper and pre-LN SSL-Large should show both → production-model natural experiment of their trained-from-scratch ablation (no training cost for us). (ii) Their first-token mechanism suggests an audio analogue: a block of N *identical* padding frames in a bidirectional encoder attends mostly within itself (identical keys), so its attention output is a near-static function — the "self-sinking" regime — predicting spikes that grow with padding length (testable dose–response). (iii) d_head is 64 in all Whisper sizes (d_model/heads: 384/6, 512/8, 768/12, 1024/16, 1280/20) → head dim fixed across the ladder; a clean control.

## E04 [2605.06611] "The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity" — ICML 2026 (arXiv comment "Accepted to ICML 2026"; OpenReview ICML'26 list)
- Claim chain (Llama-2-7B, §3–4): causal mask → first token attends only to itself (a_{0,0}=1) → no value averaging → dimension-wise variance (across a batch of *random-token* sequences) is highest at position 0 and decays with position → W_O preserves it (Kendall τ mean 0.32) → pre-FFN RMSNorm → FFN "super neurons" (large-norm gate/up columns, e.g. neuron 7890) fire selectively → heavy-tailed W_down row channels it into a few dims (e.g. 2533) → dominance ratio max/mean ≈ 262.9× after RMSNorm in layer 2 → key ≈ a fixed row of W_K → sink heads with ~100% positive QK scores.
- Causal interventions: (i) mask token k=10 so it can only attend to itself → it becomes a new sink; (ii) mean-centred variance amplification λ(o_k−μ)+μ at k=10 → sink; plain norm scaling (λ·o_k) does NOT create a sink.
- Mitigation: sigmoid attention or proposed head-wise RMSNorm suppress sinks and speed pretraining.
- **Implication:** the theory is built on the causal mask; bidirectional speech encoders (no BOS, no causal mask) are an out-of-distribution test. Prediction to test: in Whisper, are register tokens frames whose early-layer attention is near-self-only (e.g., within a block of identical padding frames), and does masking a frame to self-attention-only create a register there (their intervention, transplanted)?

## E05 [2510.08510] Luo, Fan, Wang, et al. — "To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models" — ICLR 2026 (pwc iclr-2026 tag)
- Defines sinks by a characteristic function φ ≥ τ: token norm (Darcet; ViT sinks use τ=100), mean attention received, or sink-dimension value after RMSNorm (LLM sinks, τ=20, dims {1415,2533} in Llama-2-7B) (§2 Eq.1).
- LLaVA-1.5-7B: most ViT tokens have norm < 60; 3–5 tokens/image exceed 100 and receive ≈7× more LLM attention during generation; "this correlation is not architecturally enforced" (§3.1). Mean attention per token: non-sink 0.1532%, LLM-emerged sinks 1.27%, ViT-propagated sinks 1.13%.
- Propagated ViT sinks light up DIFFERENT LLM hidden dims {982, 2494, 3263} from LLM sinks {2533, 1415}; these dims appear only after multimodal training.
- ViT sinks carry coarse global semantics (relevance maps broad; logit-lens word distributions name the main object). Sink-only input helps "global" tasks; removing sinks helps "local" tasks.
- Training-free "sink-to-the-front" gains are small (LLaVA-eval avg +0.02…+0.42; MathVista +0.4…+1.8). Training-based DIYSink (dual-MLP projector + CoT/ReW token selection) gives up to +5.8 avg on TinyLLaVA-3B.
- **Implication:** direct vision precedent for "encoder registers propagate into the LLM". Audio analogue untested: Whisper-large-v3-derived encoders feed most LALMs; whether encoder high-norm frames become audio sinks with distinct LLM dims, and what they encode (speaker/language/scene?), is open. Caveat: gains from exploiting sinks were modest in vision → for us the LALM part is best as *analysis + one application*, not the headline.

## E06 [2609.32808] Ngnawé, Pequignot, Sahoo, et al. — "Mind the Spike: Mechanisms and Brittleness of Visual Massive Activations in LVLMs" — arXiv Sept 2026; also an ICLR 2027 submission (OpenReview TPbu1cHiCa)
- Criteria matter (§2, Fig. criteria): Sun et al.'s original ratio Ψ = max spike-channel |x| / median |x| over all tokens & channels, spike if Ψ ≥ 1000 AND |x| ≥ 100; Kang et al.'s within-token ratio φ = max/RMS ≥ 20 flags 83–100% of image tokens in 21/25 LVLMs (bound φ ≤ √D) → φ is not discriminative. Sinks measured separately: token share ≥ 0.3 of attention to image tokens (Gu et al.).
- 25 LVLMs (2B–72B): 10 spike (Ψ peaks ≥ 5,278), 15 never spike (peaks ≤ 877). Spike → sink on 96.7–97.6% of spiking forwards; some non-spiking models have sinks without massive activations.
- A "visual switch" block (usually deeper than the text switch) writes the spike when a token aligns with a trigger direction t (estimated from weights by gradient ascent, recovers Sun et al.'s eigendirection, cos 0.999–1.0). Visual spike *needs attention to the text-spike sink* (blocking it suppresses 75–100%).
- **Location rule (§4):** "shared part" p_i = x_iᵀ m_{−i}/‖m_{−i}‖ (projection on the mean of the other tokens) at the decoder input: median AUC 0.893 (norm 0.850, direction 0.824); spike token in the least-shared tenth on 43–97% of images (chance 10%). Scaling a region's vectors to 1/10 draws the spike into it (33–79% vs 9–19%) in 6/10 models; "the smallest candidate takes the spike".
- Brittleness: corruptions create 3,152 vs remove 476 spikes; a trigger-guided ℓ∞ = 1/255 attack creates/removes spikes in 9/10 spiking models with yes/no answers unchanged 93–100%.
- Preventive intervention: project out t from the FFN input at the switch → spikes eliminated on clean images in 6/10 models, 1,599/1,600 POPE answers identical in 8 models.
- Compute disclosed: ≈53 B200-hours (~$360 at Sept-2026 rates).
- **Implication (key for our design):** (a) adopt Ψ (with |x|≥100 floor) + norm-ratio + sink share as *separate* observables; do not use φ. (b) Competing location rules that audio can dissociate: "least-shared / smallest token" (Mind the Spike) vs "redundant/uninformative token" (Darcet) vs "self-sinking block of identical tokens" (Anatomy). Whisper padding is simultaneously low-energy AND highly redundant; a factorial with *constant non-silent* frames (high energy, high redundancy), *low-level noise* (low energy, low redundancy) and digital silence separates them — something images cannot do cleanly.

## E07 [2609.38878] "Audio Token Attention Is Predictable Before the Language Model Runs" (Triage) — arXiv 30 Sep 2026; ICLR 2027 submission (OpenReview 1DvvJkCmz2)
- A closed-form linear map from encoder outputs predicts all-layer LLM attention to audio tokens (ρ ≥ .69 on 11/13 LALMs; fails on two Qwen-Audio models); used for pre-LLM audio-token pruning.
- Free signals are weak: **encoder state norm ‖e_i‖ correlates NEGATIVELY with LLM attention on Qwen2.5-Omni-3B (ρ −.35…−.54) and ranges −.54…+.12 across 7 LALMs** ("high-norm frames are less attended"). Opposite sign to the vision finding (ViT high-norm tokens get ~7× attention, E05).
- An audio "attention sink" (most-attended audio token) exists; dropping its column raises ρ; not characterised further. Retained tokens carry little silence (1.8% vs 15.4% of TEDLIUM frames).
- **Implication:** LALM encoders output *post-ln_post* states (Whisper's final LayerNorm), which normalises register norms away; the LALM angle must be stated as an open, secondary question (does the register's direction, not norm, mark audio sinks?). Not our headline; no one has characterised audio-encoder registers inside LALMs.

## E08 [2412.11272 v2] Wang, Xu, Lin (UVA) — "WhisperFlow: speech foundation models in real time" (MobiSys-style systems paper; TeX on arXiv is a template, read from PDF)
- Tab.1 (Whisper on LibriSpeech): pad to 30 s: 569.4 encoder GFLOPS, WER 0.048; **no padding: 152.5, 0.097; 1 s padding: 170.6, 0.111; 5 s padding: 243.8, 0.083** → shorter padding causes hallucinations (example outputs show runaway continuations). Non-monotone in padding length.
- "Hush word": a 0.5 s *raw-audio* segment, the only trainable parameter (model frozen), appended at the end and trained per model on a small dataset to reproduce the transcript → input ~10 s, encoder GFLOPS ÷3 "without major accuracy drop"; naive zero/white-noise padding of 1–5 s is 1–3 points worse than the hush word on LibriSpeech/TED-LIUM; no padding is 5–10 points worse.
- **Implication:** strongest padding-free baseline; requires per-model training. Our register account predicts *why* a learned input segment works (it provides a register/no-op target) and why naive short zero padding fails.

## E09 [2507.10860] Argmax "WhisperKit: On-device Real-time ASR with Billion-Scale Transformers" (ICML 2025 per authors)
- Self-distils Whisper-large-v3-turbo's encoder with block-diagonal attention (15-s blocks, "d750") so that a fully zero-padded 15-s block's encoder output can be precomputed once ("silence caching") and reused; block-causal variants keep accuracy but forbid silence caching. Requires training (self-distillation on Common Voice 17).
- **Implication:** the decoder in their system still sees 30 s of encoder states, half of them cached silence — consistent with the decoder needing "silence evidence". Our training-free variant must be compared against this idea (cached padding tail without block masks; tested in the pilot as `nopad+tail`).

## E10 [2607.01108] "NPUsper: Eliminating Redundant Computation for Real-Time Whisper on Mobile NPUs" — arXiv Jul 2026
- Fig. motivation: hallucinations rise for short unpadded inputs; WER/CER fall as input approaches 30 s; encoder time with padding ≈ 3.8× no-padding (≈9 s inputs, Whisper-base on NPU).
- **Key evidence:** "when hallucinated tokens generated beyond the reference transcription are removed in offline post-processing, Whisper inference without padding achieves WER and CER close to those obtained with full 30-second padding" → the unpadded failure is dominated by decoder *termination/continuation* errors, not by worse encodings of the speech frames. No-padding hallucination rate 34.4% (ablation table).
- Their fix: detect hallucinated tokens online from backward shifts in final-layer cross-attention (training-free) and stop decoding.
- **Implication:** our mechanism study must separate encoder registers from decoder end-of-speech evidence; NPUsper already shows the decoder side matters most for WER. A register paper whose *only* application is padding-free Whisper would be competing with WhisperFlow/WhisperKit/NPUsper/ICASSP-26 FT — the application must be framed as a consequence of the mechanism, not the main novelty.

## E11 [2309.16588] Darcet, Oquab, Mairal, Bojanowski — "Vision Transformers Need Registers" — ICLR 2024 (foundational; pre-2025, cite as origin only)
- DINOv2 norms are bimodal; criterion "norm > 150" = high-norm; 2.37% of tokens (§3.1). Outliers appear after ~1/3 of training, only in the three largest sizes (L, H, g) of DINOv2, "around the middle of the model".
- High-norm tokens sit on patches whose input embeddings are very similar to their 4 neighbours (redundant), hold little local information (position/pixel probes) and more global information (image-level probes).
- Hypothesis (verbatim gist): large, sufficiently trained models learn to recognise redundant tokens and use them to store, process and retrieve global information.
- §(discussion) "OpenCLIP and DeiT-III exhibit outliers both at size B and L" → the size threshold is DINOv2-specific; pretraining paradigm matters.
- Fix: N learned register tokens trained from scratch.
- **Implication:** our plan cites Darcet only as the origin of the phenomenon; the hypotheses we test come from the 2025–2026 A* papers (E01–E06, E12+).

## E12 [2608.30927] "Stride-k Subsampling: Train-Free Audio Token Reduction for Whisper" — EMNLP 2026 Main
- Whisper's fixed 1500-token encoder interface is redundant: keeping every 2nd token after the conv stem and again after the encoder (75% fewer tokens, 52–58% fewer GFLOPs) costs little WER on most benchmarks.
- §Setup facts about LALMs: **Audio Flamingo 3 and Qwen2-Audio map the *padded* mel to 1,500 conv-stem tokens, run a 32-layer Whisper-large-v3-style encoder, then AvgPool1d(2) → 750 audio tokens**; LLaMA-Omni2 uses a Whisper-large-v3-style encoder with 5-frame concatenation (1500→300).
- HF source check (transformers 5.19, this session): Qwen2-Audio and AudioFlamingo3 build a bidirectional attention mask from the feature mask → **padding frames are masked out as keys inside these LALM encoders**; Qwen2.5-Omni's audio encoder runs on valid-length windows (n_window chunks, cu_seqlens) → no padding at all. Whisper-large-v3 itself attends to padding.
- **Implication — natural experiment:** same encoder family (Whisper-large-v3 init), trained with padding visible (Whisper) vs padding masked (Qwen2-Audio, AF3) vs windowed/no padding (Qwen2.5-Omni). Prediction: registers move from padding into in-utterance silence/low-energy speech frames when padding is unavailable.

## E13 [2603.11487] "Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks" — ACL 2026 (pwc acl-2026 tag)
- Necessity theorems: for a trigger-conditional task (output the average of past tokens when a trigger appears, else a default/zero output), any softmax single-layer model with vanishing error must put attention →1 on a fixed sink (BOS) at all non-trigger positions; multi-layer: at least one layer must sink somewhere. ReLU attention solves it with no sink (constructive theorem). Intuition stated: "normalization over a probability simplex must force attention to collapse onto a stable anchor to realize a default state (e.g., when the model needs to ignore the input)."
- Theory is for causal LMs with a BOS anchor.
- **Implication:** speech is full of "ignore the input" states (silence, padding, non-speech) — a natural setting where a default/no-op anchor is needed. In a BOS-free bidirectional encoder the anchor must be *manufactured* from input frames → predicts registers on silence/padding; a register-free/ReLU-attention speech encoder is the theory-implied counterfactual (out of our compute budget; cite as future work).

## E14 [2402.17762] Sun, Chen, Kolter, Liu — "Massive Activations in LLMs" (COLM 2024; foundational, pre-2025)
- Working definition (§2): "an activation qualifies as a massive activation if its magnitude surpasses 100 and is at least or around 1,000 times larger than the median magnitude of its hidden state." They act as fixed biases; ViTs have them too (smaller).
- **Implication:** our census uses exactly this criterion (|x|>100 AND ≥1000× median) for "massive", and reports norm-ratio and sink share separately (cf. E06's warning against within-token ratios).

## E15 [2212.04356] Radford et al. — Whisper (ICML 2023; architecture facts used in the plan)
- 30-s segments; 80-ch log-mel, 25 ms window, 10 ms stride; two conv stem (2nd stride 2) → 1500 frames; sinusoidal position embeddings; **pre-activation residual blocks; final LayerNorm on encoder output**; decoder learned positions. Non-speech segments are trained (sub-sampled) with a <|nospeech|> target.
- Table "Architecture details": Tiny 4L/384/6H, Base 6/512/8, Small 12/768/12, Medium 24/1024/16, Large 32/1280/20 → **head dim = 64 for every size** (a fixed-head-dim scale ladder; E03 shows head dim drives sinks).

## E16 [2602.22394] "Vision Transformers Need More Than Registers" (LaSt-ViT) — CVPR 2026 (arXiv comment; pwc cvpr-2026)
- Defines artifacts beyond high norm: **Patch Score** = cos(patch, CLS) and **Point-in-Box (PiB)** = share of images whose top-scoring patch lies inside the foreground box. ViTs: PiB 39.8–45.3 vs ResNets 53.9–71.1; registers remove high-norm tokens but do NOT raise PiB (ViT 42.7 → 41.5; OpenCLIP 39.8 → 37.6) → "high norm is not the root cause of artifacts".
- Hypothesis "lazy aggregation": coarse (image-level) supervision + global attention make ViTs encode global semantics in background patches; bias appears from the start of training; masking top-50% scoring patches barely changes ImageNet accuracy. Evidence: larger patches / window attention raise PiB but lower accuracy.
- **Implication:** audio has a ready "PiB-in-time": AudioSet-strong / DESED event boundaries and VAD for speech. Prediction: clip-supervised encoders (AST, BEATs fine-tuned) show background (non-event) frames with the highest CLS similarity, while frame-level objectives (HuBERT/WavLM masked prediction) should not — a supervision-granularity test that vision cannot run with the same controls.

## E17 [2502.07004] "Demystifying Singular Defects in Large Language Models" (ICML 2025)
- Verified wording: "In our observations, LLaMA-like models exhibit extremely high-norm tokens … whereas BERT-like models do not"; "we did not observe high-norm tokens in bidirectional models, such as BERT, DistilBERT, and RoBERTa"; "we conjecture that the causal self-attention mechanism is one of the defining factors for the emergence of high-norm tokens."
- **Caveat for our claims:** bidirectional ViTs already have high-norm tokens (E11), so "speech refutes the causal conjecture" is NOT novel on its own. The defensible, more interesting claim: bidirectional encoders on *continuous, redundant* inputs (vision, speech) have them while bidirectional *text* encoders do not → redundancy/continuity, not causality, is the candidate factor; speech lets us manipulate redundancy directly (identical padding frames, stationary tones).

## E18 [2510.06477] Queipo-de-Llano et al. — "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin" — ICLR 2026 (pwc iclr-2026)
- Metrics (§2): matrix-based entropy H(X) from the singular values of the token-representation matrix; anisotropy p1 = σ1²/‖X‖_F²; sink score/rate of Gu et al.
- Theorem: a single massive-activation row forces spectral dominance (σ1² ≥ M) and upper-bounds H(X). Empirically BOS-norm spikes (10³–10⁴×), entropy drop (<0.5 bits) and sink-rate surge co-occur layer by layer (r(Δnorm, Δentropy) = −0.9 ± 0.18 over 6 LLMs), from ~step 1k of Pythia training.
- Causal: ablating the layer-0 MLP write to BOS in Llama-3-8B keeps entropy ~0.4–0.5 bits (vs 0.02), sink rate 0 (vs 0.85–1.0).
- "Mix–Compress–Refine": embedding tasks peak in compressed middle layers; generation needs full depth.
- **Implication (cheap, sharp test):** in speech encoders, does the register-emergence layer coincide with an entropy valley, and does the layer where probes (phone/speaker) peak sit inside it? Speech SSL layer-wise probing is a large literature (SUPERB) — a register/compression account of "which layer is best" would interest the speech community.

## E19 [2605.08504] "A Single Layer to Explain Them All: Understanding Massive Activations in LLMs" — ICML 2026
- Massive activations emerge abruptly in one "Massive Emergence (ME) layer" (pre-FFN RMSNorm concentrates the token on large-scale dims, FFN amplifies), then persist via the residual stream as a near-invariant direction; sinks appear in the next layer. A training-free mask of attention-input dims with large RMSNorm weights (from the ME layer on) improves instruction-following/math and weakens sinks.
- **Implication:** our pilot already shows a single abrupt emergence layer in Whisper (base L4/6, small L7/12, E20); the ME-layer analysis (norm-weight vs FFN contribution) is directly portable to LayerNorm-based Whisper.

## E20 [2510.22603] Anand, Cappellazzo, Petridis et al. — "Mitigating Attention Sinks and Massive Activations in AVSR with LLMs" — ICASSP 2026
- LLM-side only (Llama-AVSR: Whisper audio encoder + AV-HuBERT video encoder → avg-pool → projector → Llama-3.2-3B, LoRA). BOS sink from pretraining; intermediate sinks on low-semantic tokens (<audio>, </audio>, prompt) emerge during fine-tuning; massive activations from layer-2 MLP on shared feature indices; sink hidden states have high cosine with BOS; decorrelation loss improves WER at high downsampling.
- **Does not analyse the audio encoder's own high-norm/register frames.**

## E21 [2605.10815] "Probing Cross-modal Information Hubs in Audio-Visual LLMs" — ICML 2026 (KAIST MM, J.S. Chung's group; code kaistmm/crossmodal-hub)
- LLM-side: integrated audio-visual information concentrates in (cross-modal) sink tokens of AVLLMs (Qwen2.5-Omni, Qwen3-Omni, video-SALMONN…); sinks defined following LVLM practice, aggregated across layers; training-free hallucination mitigation by steering toward cross-modal sinks. Same group: Omni-LLM sinks (2603.14337, "OutRo").
- **Scoop-risk note:** this group is the most likely to extend sinks to audio encoders next; neither paper studies encoder-side registers, silence or padding.

## E22 OUR PILOT v1 (this session, 2026-10-10; experiments/pilot_padding/whisper_regs.py; RTX-3050; Whisper-base; LibriSpeech test-clean, 100 utts of 2–8 s; greedy decoding, max 160 tokens)
- Hand-written encoder matches HF WhisperEncoder (max |Δ| 6.9e-4 at output scale 23.6); greedy decode identical to HF generate on a test utterance.
- Reference padded run: register emergence at block 4 of 6 (max/median norm 33.9, **Ψ = 1,122, max |x| = 501 → meets Sun's massive criterion**); top-norm frames in padding (1445, 1446, 1453, 1454).
- WER / hallucination: pad30 5.39% / 1%; **nopad 385.5% / 44%** (runaway repetition); pad1 31.2%; pad2 16.3%; pad5 5.94%; pad10 5.86%.
- Appending cached padding-tail outputs for the decoder (`nopad+tail`): 24.5% (fixes most loops); transplanting 4 register KV frames into the encoder (`nopad+reg`): 315% (≈ no help); both: 25.0%.
- **Necessity test negative:** masking encoder self-attention to the utterance's 4 top-norm frames from block 4 on (`pad30-mask`): 5.39% (= baseline).
- **Interpretation (provisional, n=100, one model):** in Whisper-base, padding is used (i) by the decoder as end-of-speech/no-op evidence and (ii) by speech frames during encoding (5 s of *jointly encoded* padding ≈ full padding, while post-hoc tail = 24.5%). The 4 high-norm "register" frames alone are neither necessary nor sufficient. → The paper must not assume "registers explain padding"; it must test it (pilot v2 cached-padding-KV dose–response running).

## E23 OUR PILOT v2 — cached-silence interventions (experiments/pilot_padding/whisper_padkv.py; 2026-10-10/11; RTX-3050; LibriSpeech test-clean, same 100 utts of 2–8 s; greedy)
Cache = one encoding of a *silence-only* 30-s input: per-layer K/V and final outputs ("tail"). Speech-only encoding attends to cached K/V at positions [T,1500) ("kv"); decoder may additionally see the cached tail ("tail"). Registers = top-norm frames at the emergence block; sinks = top attention-received frames in the silence-only run.

| condition | base WER | small WER | large-v3-turbo WER | enc FLOPs (base/small/turbo) |
|---|---|---|---|---|
| pad30 | 5.28 | 2.56 | 2.02 | 1 / 1 / 1 |
| nopad | 372.8 | 214.4 | 427.8 | .115/.125/.136 |
| pad5 (real padding) | 5.82 | 2.79 | **35.07** | .253/.271/.288 |
| tail only | 5.66 | 3.96 | 22.73 | ≈nopad |
| kv only | 61.6 | 426.8 | 316.5 | .157 |
| **kv+tail** | **5.12** | **2.64** | **1.71** | .157 |
| kvTOP4+tail / kvRND4+tail | 5.04 / 5.66 | 3.88 / 3.96 | **6.83 / 21.41** | ≈nopad |
| kvTOP16+tail / kvRND16+tail | 5.43 / 5.74 | 3.80 / 3.96 | **2.64 / 17.53** | ≈nopad |
| kvTOP64+tail / kvRND64+tail | 12.49* / 5.51 | 2.95 / 3.65 | 2.40 / 10.16 | ≈nopad |
| kvTOP256+tail / kvRND256+tail | 5.20 / 5.20 | 2.33 / 3.18 | 1.94 / 2.09 | .123/.132/.140 |
| kv + decoder sees only TOP4 / TOP16 tail frames | 14.66 / 14.27 | 27.46 / 12.34 | 1.86 / 1.94 | .157 |
| pad30, top-4 / top-16 norm frames removed from decoder view | 5.35 / 5.51 | 2.56 / 2.56 | 2.64 / 2.02 | 1 |
| pad30, encoder attention to top-4 / top-16 norm frames masked from emergence block | — | 2.56 / 2.40 | 1.86 / 1.86 | 1 |
(*single unstable cell; to re-check at scale.)
- Silence-only run: register emergence block base 4/6 (35×), small 7/12 (58.5×), turbo 20/32 (29.4×, persists to the last block). Top-attended frames (sinks): base [0,1499,1453,1445,1446,1454,1452,175]; turbo [0,1496,1499,1487,1455,1495,1456,1494] → **sinks sit at the very start and at the end of the 30-s window (absolute positions).**
- **Open check:** overlap between top-norm ("register") frames and top-attended ("sink") frames per layer was not yet measured; in base the silence-only top-attended list contains the top-norm frames 1445/1446/1453/1454. Phase-0 must report both sets per layer before claiming "sinks ≠ registers".
- **Findings (provisional, n=100):** (1) high-norm *register* frames are dispensable for ASR in all three models (masking/removing them leaves WER unchanged); (2) what matters is silence used two ways — decoder end-of-speech evidence (tail) and encoder attention to a few **position-anchored sink frames**; (3) dependence grows with scale: in turbo, 16 top-attended cached frames recover 2.64% vs 17.53% for 16 random ones, and 5 s of real padding is NOT enough (35%) — *hypothesis (to test): because it never instantiates the end-of-window sink positions*; (4) training-free cached silence KV + tail matches or beats full padding at 0.157× encoder FLOPs (FLOPs analytic; wall-clock not yet measured).
- Ties to theory: sinks ≠ massive activations (E03 decoupling) shown in a production speech model; default-state/no-op anchor (E13) realised by silence; position anchoring echoes first-token sinks (E03/E04) but at the *end* of a bidirectional window.
