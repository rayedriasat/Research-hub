# Research Plan: Do Audio Transformers Need Registers?

**Working title:** *Do Audio Transformers Need Registers? Characterizing and Removing High-Norm Attention Artifacts in Pretrained Audio and Speech Models*

**Author:** (you) · **Status:** Proposal draft for supervisor review · **Date:** 2026-07-15
**Target venues (in order):** ICASSP 2027 → Interspeech 2027 → IEEE/ACM TASLP (Q1 journal fallback)
**Compute envelope:** Free Kaggle GPU (T4/P100, 30 h/week) for ~90% of the work; ~10–30 A100-hours for the optional distillation baseline and ablations.

---

## 0. One-paragraph summary (the elevator pitch for your supervisor)

A landmark ICLR 2024 paper ("Vision Transformers Need Registers", Darcet et al.) discovered that large pretrained ViTs silently repurpose ~2% of their input tokens as internal "scratch space": these tokens develop 10× higher norms, corrupt attention maps, and hurt any downstream task that needs clean per-token features. A NeurIPS 2025 spotlight follow-up ("Vision Transformers Don't Need Trained Registers", Jiang et al.) traced the phenomenon to a sparse set of "register neurons" and showed the artifacts can be removed **training-free** at inference by shifting the outlier activations into an extra untrained token. This artifact/register phenomenon has since been studied in vision, diffusion models, LLMs, and VLMs — **but never systematically in audio**. Audio transformers (AST, BEATs, Audio-MAE, EAT, HuBERT, WavLM, Whisper's encoder) are architecturally ViTs or ViT-like encoders operating on spectrograms or waveform frames, and audio has a natural analog of "uninformative background patches": **silence and stationary noise frames**. We propose the first systematic study of high-norm artifact tokens across 6–8 pretrained audio transformers, an audio port of the training-free test-time register fix, and a downstream evaluation on standard audio benchmarks (ESC-50, Speech Commands V2, FSD50K, SUPERB-style probes, and localization-sensitive sound event detection). Every phase is inference-dominated: the whole project fits in free Kaggle GPU quota, with a small optional A100 budget for a self-distillation baseline. The paper has a high floor: *whatever* we find (artifacts present / absent / modality-dependent) is a publishable characterization, because the audio community has not yet asked this question.

---

## 1. Background: the register-token literature, paper by paper

This section explains each source paper in depth — what it did, what it found, what it used, and what gap it leaves that we exploit. TeX sources for all six are downloaded under `Papers_Tex_Source/` (with PDFs in `Papers_pdf/`), so you can read the exact method sections offline.

### 1.1 ANCHOR #1 — "Vision Transformers Need Registers" (Darcet, Oquab, Mairal, Bojanowski — FAIR/Meta & Inria)

- **arXiv:** https://arxiv.org/abs/2309.16588 · **Venue:** ICLR 2024 (oral, outstanding paper award) · **Local copy:** `Papers_Tex_Source/01_...2309.16588v2_TeX_Source/flattened_main.tex`
- **Role in our project:** The *phenomenon paper*. Defines the artifact, the diagnostics, and the analysis protocol we will replicate on audio models.

**What they discovered.** In feature maps of both supervised (DeiT-III), text-supervised (OpenCLIP) and self-supervised (DINOv2) ViTs, a small fraction of tokens (~2%) develop output norms ~10× higher than the rest. Precise findings, each with the experiment behind it:

1. **Artifacts are high-norm outlier tokens.** The token-norm distribution at the last layer (before final LayerNorm) is *bimodal*; they hand-pick a cutoff (norm > 150 for DINOv2-g) and study the two populations. → *Our audio analog: plot per-token norm histograms for every audio model; check bimodality; pick per-model cutoffs (or the 98th-percentile rule from Paper #5).*
2. **Outliers appear in the middle layers** (~layer 15 of the 40-layer DINOv2-g), **only after sufficiently long training**, and **only in sufficiently large models** (Large/Huge/giant exhibit them; Tiny/Small/Base mostly don't). → *Our audio analog: layer-wise norm tracking; compare Base vs Large variants of HuBERT/WavLM/Whisper.*
3. **Outliers appear on redundant patches.** Measured by cosine similarity of a patch (right after patch embedding) with its 4 spatial neighbors: high-norm tokens sit on patches nearly identical to their neighbors — uniform backgrounds, sky, walls. → *Our audio analog — this is our headline scientific question: do audio artifacts sit on **silence / stationary-noise frames**, the acoustically redundant regions? We can quantify redundancy exactly the same way (neighbor cosine similarity) AND with an audio-native measure (frame energy / spectral flatness), giving us a bonus analysis vision couldn't do.*
4. **Outlier tokens lose local information but gain global information.** Two linear probes: (a) *position prediction* (predict the token's own grid position from its embedding — outliers do worse) and (b) *pixel reconstruction* (reconstruct the patch — outliers do worse); plus (c) *image classification from a single token* (outliers do much better than normal tokens). → *Our audio analog: (a) predict a token's time-frequency position; (b) reconstruct its spectrogram patch; (c) classify the clip's label from a single token.*
5. **The fix: registers.** Append a few learnable, input-independent tokens to the sequence during *training from scratch*; discard them at output. Artifacts vanish; dense-task performance improves; object discovery (LOST) is repaired. → *Retraining from scratch is infeasible for us (and for most labs) — which is exactly why Anchors #2 and #3 exist.*

**Datasets/benchmarks they used:** ImageNet-1k/22k classification, ADE20k segmentation (mIoU), NYUv2 depth (RMSE), VOC07/VOC12/COCO20k object discovery via LOST (corloc).

**The gap we exploit:** The paper studies vision only. Its central mechanistic claim — models recycle *low-information tokens* as computation scratch space — makes a **testable prediction in audio**: artifacts should concentrate in silence and stationary noise. Nobody has tested this. Audio also offers naturally *variable-length* inputs and *asymmetric axes* (time vs frequency, unlike x/y in images) — both let us probe the phenomenon in régimes vision cannot.

### 1.2 ANCHOR #2 — "Vision Transformers Don't Need Trained Registers" (Jiang, Dravid, Efros, Gandelsman — Berkeley)

- **arXiv:** https://arxiv.org/abs/2506.08010 · **Venue:** NeurIPS 2025 **(spotlight)** · **Code:** https://avdravid.github.io/test-time-registers · **Local copy:** `Papers_Tex_Source/02_...2506.08010v5_TeX_Source/flattened_main.tex`
- **Role in our project:** The *method paper*. This is the method we port to audio. Its public code is our implementation starting point.

**What they discovered.** The mechanism *behind* Darcet et al.'s artifacts:

1. **Outliers appear right after a specific MLP.** Tracking max patch norm after every attention/MLP block over 1000 ImageNet images, outliers appear after the MLP of layer 6 in OpenCLIP ViT-B/16 (and attention sinks appear in the layers after). → *We reproduce this exact plot per audio model: max token norm after every block, averaged over ~1000 audio clips.*
2. **A sparse set of "register neurons" controls outlier placement.** ~5–10 MLP hidden units in the pre-outlier layers activate consistently and exclusively at outlier positions across images. Their `FindRegisterNeurons` algorithm (Alg. 1 in the paper — very simple, ~20 lines): run M images; find outlier token positions (norm > threshold); average every neuron's activation at those positions; return the top-k neurons by average activation. For OpenCLIP they use `top_layer=5`, threshold 75, `top_k=10`.
3. **These neurons are causal.** Copying a register neuron's max activation into an arbitrary token position (and zeroing it elsewhere) *moves* the outlier there — they literally draw a heart shape out of outlier tokens. Intervening on random neurons does nothing. **Important negative control:** simply *zeroing* register-neuron activations (instead of moving them) drops OpenCLIP zero-shot ImageNet by ~15% — the outlier computation is load-bearing and must be relocated, not deleted.
4. **The fix — test-time registers:** append one extra zero-initialized token; at every register neuron, copy the max patch activation into that token and zero it in the image patches; resume the forward pass. Results: ImageNet linear probe unchanged (86.4), ADE20k mIoU 48.3→49.1, NYUv2 RMSE 0.388→0.378 — matching the *retrained-with-registers* model. LOST object discovery on DINOv2: +21 corloc, within ~0–2 of trained registers. Zero-shot segmentation mIoU improves. Linear-probing the test-time register itself shows it holds global image info just like a trained register.

**Datasets/benchmarks they used:** ImageNet (linear probe + zero-shot), ADE20k, NYUv2, ImageNet-segmentation (zero-shot seg from attention maps: mIoU/pixel-acc/mAP), VOC07/12 + COCO20k (LOST corloc), VLM attribution maps, typographic attack defense.

**The gap we exploit:** Method demonstrated only on CLIP-family and DINOv2 vision encoders. The authors themselves list "other modalities" as future work in their discussion. Applying `FindRegisterNeurons` + test-time registers to AST/BEATs/HuBERT/Whisper is unexplored, and audio-specific questions arise immediately: Do register neurons exist in audio models at all? Are they in comparable relative depths? Does one register suffice, or do longer variable-length sequences need more? Does the fix improve *frame-level* tasks (the audio analog of dense prediction) more than clip-level tasks, mirroring vision?

### 1.3 ANCHOR #3 — "Vision Transformers with Self-Distilled Registers" (PH-Reg)

- **arXiv:** https://arxiv.org/abs/2505.21501 · **Venue:** NeurIPS 2025 · **Local copy:** `Papers_Tex_Source/03_...2505.21501v3_TeX_Source/flattened_main.tex`
- **Role in our project:** The *training-based baseline* we compare the training-free method against (our only phase that needs A100 time, and it is optional).

**What they did.** Post Hoc Registers (PH-Reg): add register tokens to an *existing* pretrained ViT via cheap self-distillation, no labels and no full retraining:

1. **Teacher = frozen original model + test-time-augmentation denoising.** Key trick: artifacts are *not equivariant* — shift the image and the artifacts don't shift with it. So they average dense features over ~n augmented views (shifts/flips, inverse-warped back), which cancels artifacts. No gradients needed; ~200 ms per image (~100× faster than the neural-field denoiser DVT it replaces).
2. **Student = same weights + m randomly-initialized register tokens.** Only the registers and a small unlocked subset (positional embeddings, patch conv, or last block — they ablate) are trained, with loss `1 − cos(F_t, F_s) + MSE(F_t, F_s)` on dense features.
3. **Results:** consistent gains on open-vocabulary segmentation (8 benchmarks, e.g. VOC21 59.6→63.0 mIoU over SCLIP-class baselines), linear-probe segmentation and depth, across CLIP/OpenCLIP/DFN-CLIP/DINOv2.

**The gap we exploit:** Vision-only again. But more importantly for us, PH-Reg defines the *fair comparison* our reviewers will expect: "training-free edit vs. cheap post-hoc training." Porting the denoising idea to audio is also independently interesting: the audio analog of image-shift augmentation is **time-shift** (circularly shift the waveform/spectrogram), and checking whether audio artifacts are similarly non-equivariant under time shifts is itself a novel diagnostic. **Caution:** frequency-shift is *not* a label-preserving augmentation in audio the way vertical flip is in vision — only shift along time. This asymmetry is a nice discussion point for the paper.

### 1.4 SUPPORTING #4 — "Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment"

- **arXiv:** https://arxiv.org/abs/2603.25803 (Mar 2026, reproducibility study) · **Local copy:** `Papers_Tex_Source/05_...2603.25803v1_TeX_Source/flattened_main.tex`
- **Role in our project:** The *methodological template*. This recent paper is literally the shape of our Part 1 — "take the register claims, test them across architectures" — except across *vision* architectures (DINO, DINOv2, OpenCLIP, DeiT3, plus hierarchical PVTv2/Swinv2). We do it across *audio* models. Its existence proves reassessment/characterization papers in this niche are publishable *right now*.

**What to copy from it (their exact protocol, worth adopting verbatim):**
- Terminology hygiene: they carefully distinguish **attention maps** (CLS→patch attention, last layer, head-averaged), **feature maps** (L2 norm of output tokens before final LayerNorm), **high-norm/outlier tokens** (per-model 98th-percentile cutoff — better than a hand-picked absolute threshold), and **artifacts** (anomalous patterns in either map). Reviewers punished the original paper's looseness here; we adopt the clean definitions from day 1.
- Their finding that **some claims don't generalize across architectures** (e.g., models without a CLS token need average-pooling workarounds; smaller models sometimes show artifacts too) tells us to expect — and *value* — heterogeneous results across audio models. Heterogeneity is a finding, not a failure.
- Their probe setup: position prediction = linear layer, CE + 0.5×MSE loss, Adam, cosine schedule, ≤30 epochs, early stopping patience 3; patch reconstruction = linear layer, MSE; classification = linear probe on single tokens, 200k images. We reuse these hyperparameters for our audio probes to stay comparable.
- They ran everything at a reproducibility-study scale on a *single H100 slice* (~150 GPU-hours total) — evidence our compute envelope is realistic, since our models are smaller than DINOv2-g.

### 1.5 SUPPORTING #5 — "Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs"

- **arXiv:** https://arxiv.org/abs/2510.22603 (Oct 2025, ICASSP-style short paper) · **Local copy:** `Papers_Tex_Source/04_...2510.22603v3_TeX_Source/flattened_main.tex`
- **Role in our project:** The *closest related work in audio* — we must cite it and clearly differentiate. It is our proof that the audio community is receptive to this exact topic, and our proof the niche is otherwise empty (it cites zero prior audio-encoder artifact work because none exists).

**What they did:** Analyzed **Llama-AVSR** — a *decoder LLM* (Llama 3.2-3B) fine-tuned with LoRA to consume Whisper/AV-HuBERT features for speech recognition. Found attention sinks and massive activations (features ≥10³× median) at BOS and at low-semantic intermediate tokens (`<audio>`, `</video>`, prompt tokens); showed sinks emerge *during fine-tuning*, originate in layer-2 MLP GLU gating, and that intermediate sink tokens have high cosine similarity with BOS. Fix: a decorrelation loss during fine-tuning that reduces BOS-similarity; improves WER under aggressive feature compression.

**How we differ (write this into the paper's related-work section):**
1. They study the **LLM decoder**; the audio signal enters only as projected embeddings. We study the **audio encoders themselves** (AST/BEATs/HuBERT/WavLM/Whisper-encoder/Audio-MAE) — the models the entire audio field uses as feature extractors.
2. Their phenomenon is the *StreamingLLM/BOS-sink* family (causal decoder sinks); ours is the *Darcet register-token* family (bidirectional encoder artifacts). Related but mechanically distinct (their sinks anchor on special tokens; ViT artifacts anchor on redundant *content* positions).
3. Their fix requires fine-tuning (a loss term); our primary method is training-free.
4. They evaluate one task (WER); we evaluate a benchmark suite across many audio tasks.

**Useful technique to borrow:** their massive-activation criterion (feature magnitude ≥ τ·median, τ=10³) is a second, norm-independent artifact diagnostic we can add to our analysis battery, and their MLP-GLU gating analysis is a template for our "where do outliers originate" section on models with GLU MLPs (WavLM, BEATs variants).

### 1.6 SUPPORTING #6 — "AST: Audio Spectrogram Transformer" (Gong, Chung, Glass — MIT)

- **arXiv:** https://arxiv.org/abs/2104.01778 · **Venue:** Interspeech 2021 · **Local copy:** `Papers_Tex_Source/06_...2104.01778v3_TeX_Source/flattened_main.tex`
- **Role in our project:** Our *bridge model* and the reason the vision→audio transfer is nearly frictionless.

**Key facts (from the TeX source):** AST converts t seconds of audio into a 128-band log-mel spectrogram (25 ms Hamming window, 10 ms hop → 128×100t "image"), splits it into **16×16 patches** (with overlap 6), linearly projects to 768-d, prepends a `[CLS]` token, adds positional embeddings, and runs a standard 12-layer/87M-param ViT — **initialized from ImageNet-pretrained DeiT weights** (channel-averaged patch embedding, bilinear-interpolated positional embeddings). Benchmarks: AudioSet 0.485 mAP, ESC-50 95.6%, Speech Commands V2 98.1%.

**Why this matters strategically:** AST *is* a ViT — same patch size, same width, same depth as the models where artifacts are proven, and it even starts from vision weights. If artifacts exist anywhere in audio, AST is where we'll find them first; if AST is artifact-free despite its ViT ancestry, that contrast (what about audio training removes them — shorter training? smaller data? overlap in patches? the mel input statistics?) is itself a strong finding. Either branch feeds the paper.

### 1.7 Adjacent work found in the 2026-07-16 verification sweep (cite & differentiate)

A fresh web + arXiv + citation-graph sweep (all 912 citers of Anchor #1 and all 39 citers of Anchor #2 scanned) confirmed the gap is still open but surfaced five papers to add to related work before a reviewer does:

1. **"Online Register for Dual-Mode Self-Supervised Speech Models" (arXiv:2602.23702, Feb 2026)** and its follow-up **"Online Predictive Coding for Dual-Mode SSL Speech Models" (arXiv:2606.21268, Jun 2026)** — the first papers to use the word "register" in speech models, and they cite Darcet et al. **They do not scoop us:** their registers are learnable placeholders for *missing future context* in streaming ASR — a training-time architectural device with no artifact analysis, no high-norm phenomenon, no study of pretrained encoders. **Wording consequence:** we can no longer claim "registers have never appeared in speech models"; claim instead that registers have been used as an architectural tool in speech, but nobody has asked whether pretrained audio encoders *develop the artifacts registers were invented to fix*.
2. **"Silence is Sweeter Than Speech" (arXiv:2205.03759, 2022)** — HuBERT stores *speaker* information at silence positions in the representation. Predates the register literature and is direct motivating evidence for RQ2 (models repurpose low-information silence frames for global storage). Cite in the introduction — it strengthens our hypothesis.
3. **XLSR-Transducer (arXiv:2407.04439, 2024)** — exploits attention sinks as an engineering trick (halving left context) in a streaming speech encoder. A tool, not an analysis.
4. **"Outlier Reduction with Gated Attention for Post-Training Quantization in Speech Foundation Models" (arXiv:2406.11022)** — documents sink-like no-op outlier behavior in Whisper, but in the **decoder** (the `<|transcribe|>` token), for quantization. Closest artifact evidence in Whisper; differentiate on encoder-vs-decoder and characterization-vs-quantization.
5. **"Probing Cross-modal Information Hubs in Audio-Visual LLMs" (arXiv:2605.10815, 2026)** — sink tokens store cross-modal info in AV-LLMs; same LLM-decoder family as §1.5, same differentiation applies.

Also relevant to the risk framing: **"Registers Matter for Pixel-Space Diffusion Transformers" (arXiv:2605.16147)** and **"Taming Outlier Tokens in Diffusion Transformers" (arXiv:2605.05206)** found DiTs *don't* exhibit high-norm patch outliers — the phenomenon is architecture/training-dependent, which supports our "heterogeneity is a finding, not a failure" position (§2 branch (b)/(c)).

### 1.8 Additional models to include in the study (with checkpoints)

| Model | arXiv | Params | Tokenization | Pretraining | HuggingFace checkpoint |
|---|---|---|---|---|---|
| AST | [2104.01778](https://arxiv.org/abs/2104.01778) | 87M | 16×16 spectrogram patches | Supervised (AudioSet) + ImageNet init | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| BEATs | [2212.09058](https://arxiv.org/abs/2212.09058) | ~90M | 16×16 spectrogram patches | SSL, acoustic tokenizers (iter3) | `microsoft/...` / official repo ckpts |
| Audio-MAE | [2207.06405](https://arxiv.org/abs/2207.06405) | 86M | 16×16 spectrogram patches | SSL, masked autoencoding | official FB repo checkpoint |
| EAT | [2401.03497](https://arxiv.org/abs/2401.03497) | 88M | spectrogram patches | SSL, bootstrap (data2vec-style) | `worstchan/EAT-base_epoch30_pretrain` |
| HuBERT Base/Large | [2106.07447](https://arxiv.org/abs/2106.07447) | 95M/317M | 20 ms waveform frames (CNN frontend) | SSL, masked cluster prediction | `facebook/hubert-base-ls960`, `-large-ll60k` |
| WavLM Base+/Large | [2110.13900](https://arxiv.org/abs/2110.13900) | 95M/317M | 20 ms waveform frames | SSL + denoising | `microsoft/wavlm-base-plus`, `-large` |
| Whisper encoder (small→large-v3) | [2212.04356](https://arxiv.org/abs/2212.04356) | 88M–637M (enc.) | 20 ms mel frames, conv downsampled | Weak supervision, 680k h | `openai/whisper-small/medium/large-v3` |

This grid is deliberately structured to support **causal-ish comparisons**:
- **Patch-based (AST/BEATs/Audio-MAE/EAT) vs frame-based (HuBERT/WavLM/Whisper)** — does 2D-patch tokenization inherit the ViT artifact pattern while 1D frame sequences behave like text models?
- **Supervised (AST, Whisper) vs SSL (rest)** — Darcet et al. found DINO(v1) artifact-free while DINOv2 and supervised models weren't; is there an audio analog?
- **Base vs Large within a family (HuBERT, WavLM, Whisper)** — tests the "only large models develop artifacts" claim (Claim 2 in Paper #5's taxonomy).
- **ImageNet-initialized (AST) vs audio-native (others)** — could artifacts be *imported* from vision weights? A genuinely fun question no vision paper could ask.

---

## 2. The research gap, stated precisely

**Verified empty (arXiv API keyword sweeps, 2026-07-14/15, re-verified 2026-07-16 with web search + citation-graph scan; see §9 for the queries):**
- No paper analyzes high-norm artifact tokens / register tokens in *any* pretrained audio encoder (AST, BEATs, HuBERT, WavLM, Whisper encoder, Audio-MAE, EAT).
- `"register tokens" AND cat:cs.SD` → 1 tangential hit (CAV-MAE Sync, audio-visual MAE, uses registers as a design choice, no artifact analysis). `"attention sink" AND cat:eess.AS` → 4 hits, only the Llama-AVSR paper (§1.5) relevant, and it targets the LLM decoder. `"massive activations" AND audio` → same single paper. `HuBERT AND "high-norm"` → zero.
- Citation-graph check (2026-07-16): of the **912 papers citing Anchor #1**, only ~16 are audio/speech-titled and none characterizes artifacts in pretrained audio encoders (they use registers as an ingredient, or study LLM-side sinks). Of the **39 papers citing Anchor #2** (the method we port), zero are audio. The 2026 attention-sink survey (arXiv:2604.10098) catalogs sinks in LLMs, ViTs and VLMs — audio encoders are absent as a category.
- **Wording caveat (new, from §1.7):** the "Online Register" streaming-ASR papers (2602.23702, 2606.21268) now use registers *as an architectural tool* in speech. Our claim must be scoped to *artifact characterization in pretrained audio encoders*, which remains untouched.
- Meanwhile the vision side of this niche is *hot*: registers papers in 2025–2026 for face recognition, diffusion transformers, VGGT, Mamba, VLM hallucination (YARD), plus two NeurIPS 2025 papers and a 2026 cross-architecture reassessment; registers are in production in DINOv3. Hot adjacent field + untouched modality = the definition of a blue-ocean transfer with a receptive reviewer pool.

**The five research questions:**

- **RQ1 (existence & prevalence):** Do high-norm outlier tokens / attention-map artifacts exist in pretrained audio transformers? In which models, at what layer depths, at what rates, and with what norm ratios?
- **RQ2 (localization — the audio-native question):** Where do artifacts sit in the input? Specifically, do they concentrate on acoustically redundant regions — silence, low-energy frames, stationary noise — as the vision "redundant patch" theory predicts? (Measurable: frame energy, spectral flatness, neighbor cosine similarity vs outlier status.)
- **RQ3 (information content):** Do audio outlier tokens lose local information (time-frequency position, patch reconstruction) and gain global information (clip-level class), mirroring vision?
- **RQ4 (the fix):** Can `FindRegisterNeurons` + test-time registers be ported to audio models that exhibit artifacts? Does the training-free fix clean attention/feature maps and preserve or improve downstream performance? How does it compare to a PH-Reg-style self-distilled register (optional, A100)?
- **RQ5 (downstream consequences):** Which tasks benefit? Hypothesis (from vision): clip-level classification is barely affected, but *frame-level / localization-sensitive* tasks (sound event detection with timestamps, keyword localization, diarization-style probes) improve — the audio analog of "dense prediction improves, classification doesn't."

**Answer-independence (why the paper can't fail):** RQ1 has three possible outcomes and all are publishable: (a) artifacts everywhere → full transfer story + fix; (b) artifacts in some models only → the contrast (patch vs frame? supervised vs SSL? big vs small?) is the headline; (c) artifacts nowhere → "Audio Transformers Don't Need Registers" — a surprising negative result requiring an explanation (training length? data scale? mel statistics?), publishable with careful analysis, and the typographic-attack-style stress tests in §4.3 give us fallback content.

---

## 3. Methodology in detail

### Phase A — Artifact census (RQ1, RQ2) — Weeks 1–4, free Kaggle GPU

**Data for analysis passes (no training, just forward passes with hooks):**
- ~2000 clips from **AudioSet eval** or **FSD50K eval** (general sound events, mixtures, music, speech),
- ~1000 utterances from **LibriSpeech test-clean** (clean read speech, natural leading/trailing silence),
- ~500 clips from **ESC-50** (environmental sounds, 5 s each),
- a **synthetic diagnostic set** you generate yourself: pure silence, white/pink noise, tones, clicks, speech+silence concatenations at controlled SNRs. This set is our precision instrument for RQ2 and costs nothing.

**Instrumentation (PyTorch forward hooks on every block):** for each model and clip record
1. per-token L2 norm after every transformer block (before final LayerNorm — Paper #5's convention);
2. attention maps: CLS→token (patch models) and mean-received-attention per token (frame models, which lack CLS — use Paper #5's average-pooling workaround);
3. attention entropy per head per layer;
4. massive-activation index sets (τ·median criterion, τ=10³, from Paper #6/§1.5);
5. MLP hidden activations for the pre-outlier layers (needed later for `FindRegisterNeurons`).

**Analysis criteria — the numbers that decide RQ1 (pre-register these thresholds before running, it makes the paper stronger):**
- *Outlier definition:* token norm > 98th percentile of that model's pooled norm distribution (Paper #5), plus a bimodality check (e.g., Hartigan's dip test or simple GMM-vs-1-component BIC) on the distribution. **Call a model "artifact-positive" only if** the distribution is bimodal AND the top mode is ≥3× the median AND outlier positions attract disproportionate attention (mean received attention ≥5× uniform share). Otherwise "artifact-negative." Report the continuous quantities regardless so borderline models are visible.
- *Layer profile:* max-token-norm-vs-layer curves averaged over ≥1000 clips (Anchor #2's Fig. 1 protocol) — identifies the emergence layer.
- *RQ2 localization metrics:* for every token, compute (i) frame/patch log-energy, (ii) spectral flatness, (iii) cosine similarity to spatial neighbors after the patch/frame embedding (Darcet's redundancy measure — 4-neighborhood for 2D patch models, 2-neighborhood along time for frame models). Then report AUROC of "is-outlier" against each covariate + point-biserial correlations. The synthetic diagnostic set gives the clean causal version: concatenate 2 s speech + 2 s silence and ask *where the outliers land*.

**Deliverable of Phase A:** a results table (model × {artifact-positive?, % outlier tokens, norm ratio, emergence layer, silence-AUROC}) + a figure gallery (spectrogram / norm-map / attention-map triptychs). This alone is a workshop paper; keep going.

### Phase B — Information probes (RQ3) — Weeks 4–6, free Kaggle GPU

Replicate the three probes from Anchor #1 / Paper #5 with their exact hyperparameters (linear heads; CE+0.5·MSE for position; MSE for reconstruction; Adam/AdamW, cosine schedule, ≤30 epochs, early stop patience 3):
1. **Position prediction:** from a token embedding, predict its (time, frequency) grid position (patch models) or time index bucket (frame models). Compare outlier vs normal token accuracy.
2. **Patch reconstruction:** linear head reconstructs the 16×16 log-mel patch (or the frame's mel slice). MSE by token type.
3. **Global-info probe:** linear classifier on a *single token* (CLS vs random-normal vs random-outlier) predicting the clip label on ESC-50/Speech Commands. If outlier tokens beat normal tokens, the "global aggregation" theory transfers.

Feature extraction dominates the cost (one pass over each probe dataset, cache to disk); the probes themselves train in minutes, even on CPU.

### Phase C — The fix: test-time registers for audio (RQ4) — Weeks 6–9, free Kaggle GPU

For each artifact-positive model:
1. Port `FindRegisterNeurons` (Anchor #2 Alg. 1 — their code is public): find outlier positions on ~1000 clips, average MLP activations there, take top-k (k=10 to start; ablate 5/10/20). Search layers up to the emergence layer found in Phase A.
2. Validate causality exactly as they did: *move* an outlier to a chosen frame (copy max activation in, zero elsewhere) and confirm norms/attention follow. Include the two negative controls: random-neuron intervention (should do nothing) and zeroing-without-moving (should hurt accuracy — measure the drop on ESC-50 zero-shot/probe).
3. Add the test-time register: one extra zero-init token, shift outlier activations into it at each register neuron, run the model. Ablate 1/2/4 registers — audio sequences for 10 s clips are ~1200 tokens (AST) vs ViT's 196, so audio may genuinely need more registers; sequence-length dependence is a novel audio-specific ablation, run it on 1 s / 5 s / 10 s / 30 s inputs.
4. Confirm artifact removal: re-run the Phase A battery post-edit; before/after attention maps are the money figures.

**Optional Phase C′ (the A100 piece, ~10–30 hours):** PH-Reg-style baseline on the 1–2 most artifact-positive models. Teacher denoising via **time-shift** test-time augmentation only (no frequency shifts — not label-preserving; document this asymmetry). Student = +m registers, unfreeze registers + positional embeddings (their best cheap setting), cosine+MSE distillation loss on unlabeled AudioSet clips. This gives the "training-free vs cheap-training" comparison table reviewers will want. Skip or defer if Phase C results are already strong — state it as future work instead.

### Phase D — Downstream benchmarks (RQ5) — Weeks 9–12, free Kaggle GPU

Frozen-feature linear/shallow probes (HEAR/SUPERB-style — never fine-tune the backbone; cache features once, train heads in minutes), each run **before vs after** the test-time-register edit:

| Task | Dataset | Metric | Why it's in the suite |
|---|---|---|---|
| Environmental sound classification | **ESC-50** (2k clips, 50 classes, 5-fold) | accuracy | Tiny, standard, AST's own benchmark |
| Keyword spotting | **Speech Commands V2** (105k 1-s clips, 35 words) | accuracy | Clip-level speech; AST benchmark |
| Sound event tagging | **FSD50K** (51k Freesound clips, 200 classes) | mAP | Multi-label, AudioSet-eval stand-in without the YouTube download pain |
| Speaker identification | **VoxCeleb1** probe (SUPERB SID protocol) | accuracy | Non-semantic speech attribute |
| Emotion recognition | **CREMA-D** (7.4k clips, 6 emotions) | accuracy | Paralinguistic; small |
| **Sound event detection (frame-level)** | **DESED / MAESTRO-real subset** or strong-labeled AudioSet subset | event-F1 / PSDS | **The localization-sensitive task where we predict the biggest gain** — audio's "dense prediction" |
| Zero-shot "attention-as-localization" | strong-labeled clips: binarize attention along time, score against event boundaries | IoU/mAP | Direct analog of Anchor #2's zero-shot segmentation — cheap and very visual |

Also report the **null results honestly**: vision saw ~0 change on classification; we expect the same on ESC-50/KWS, and *saying so* is part of the story ("the edit is safe: no regression on N tasks, gains where token-level fidelity matters").

**Statistics:** 3 seeds per probe, mean±std, 95% CIs (Anchor #2 reports CIs; Paper #5 fixed seed 42 — we do better), paired bootstrap over clips for before/after deltas.

---

## 4. Paper skeleton, title, abstract, keywords

### 4.1 Draft abstract (for the supervisor meeting — rewrite after results)

> High-norm "artifact" tokens are a well-documented pathology of pretrained Vision Transformers: a small set of tokens is silently repurposed for global computation, corrupting attention maps and degrading dense downstream tasks, with learned or test-time "registers" as the established remedy. Whether this phenomenon extends to audio transformers is unknown. We present the first systematic study of attention artifacts across N pretrained audio and speech encoders spanning spectrogram-patch models (AST, BEATs, Audio-MAE, EAT) and waveform-frame models (HuBERT, WavLM, Whisper), supervised and self-supervised objectives, and multiple model scales. We find [artifacts emerge in …], concentrating on [silence and acoustically redundant frames], [losing local time-frequency information while encoding clip-level semantics]. We port the training-free test-time register method to audio, showing it [removes artifacts and improves frame-level tasks such as sound event detection by …] without any retraining, while leaving clip-level accuracy intact. Our results [extend/qualify] the token-recycling account of transformer artifacts beyond vision, and provide a drop-in diagnostic and fix for the frozen audio encoders used across the field. Code and analysis tooling released.

### 4.2 Keywords

`audio transformers` · `register tokens` · `attention artifacts` · `high-norm tokens` · `attention sinks` · `massive activations` · `self-supervised speech models` · `audio spectrogram transformer` · `interpretability` · `training-free model editing` · `sound event detection` · `representation analysis`

### 4.3 Section plan

1. Introduction (the vision story in 2 paragraphs; the audio gap; contributions C1 census, C2 localization law, C3 training-free fix, C4 benchmark suite + code)
2. Related work (registers in vision — Anchors 1–3 + reassessment #5; attention sinks in LLMs — StreamingLLM, massive activations; the audio-adjacent work — §1.5 Llama-AVSR sinks, §1.7 streaming "online registers", Whisper-decoder quantization outliers, silence-as-speaker-storage — and how we differ; audio transformer families)
3. Preliminaries: models, tokenizations, terminology (adopt Paper #5's four clean definitions verbatim)
4. Artifact census (Phase A) 5. Information probes (Phase B) 6. Test-time registers for audio (Phase C/C′) 7. Downstream impact (Phase D) 8. Discussion & limitations
- Stretch appendix if time permits: audio "typographic attack" analog — does an artifact-bearing silence region make models more prone to ignoring brief events? (Anchor #2 has a typographic-attack appendix we can mirror conceptually.)

---

## 5. Compute & storage budget (say these numbers to your supervisor)

| Phase | GPU need | Estimate |
|---|---|---|
| A census | inference w/ hooks, 6–8 models × ~4k clips | 15–25 T4-hours |
| B probes | 1 extraction pass + linear heads | 10–15 T4-hours |
| C fix | neuron search + ablations (inference) | 10–20 T4-hours |
| C′ PH-Reg baseline (optional) | distillation, registers+posemb only, 1–2 models | **10–30 A100-hours** (~$20–60) |
| D benchmarks | feature extraction ×2 (pre/post edit), heads are trivial | 15–25 T4-hours |
| **Total** | | **~60–90 free Kaggle hours over ~12 weeks + optional small A100 budget** |

Kaggle free tier = 30 GPU-h/week → never the bottleneck. Storage: cached features are the main cost (~50–200 GB across probe datasets; prune per-layer caches to the layers you need). All models fit fp16 inference in <8 GB VRAM except Whisper-large-v3 encoder (~4 GB fp16 — also fine).

## 6. Week-by-week schedule (12 weeks)

| Week | Milestone |
|---|---|
| 1 | Environment + data plumbing. Load all checkpoints, run one clip through each, plot spectrograms + attention maps. *Learn to read mel spectrograms (one day: time × log-frequency image; silence = dark horizontal band-free regions; speech = harmonic stacks).* Reproduce Anchor #2's norm-vs-layer plot on OpenCLIP with their public code to validate your instrumentation. |
| 2–3 | Phase A on all models, natural data. First verdict table + figure gallery. **Checkpoint: which models are artifact-positive?** |
| 4 | Phase A synthetic diagnostics (silence/noise/SNR sweeps) → RQ2 answer. |
| 5–6 | Phase B probes. Mid-project writeup (becomes paper §4–5). Show supervisor. |
| 7–8 | Phase C port: FindRegisterNeurons, causal moving, negative controls, before/after maps. |
| 9 | Register-count & sequence-length ablations. Decision point on C′ (A100). |
| 10–11 | Phase D benchmark suite pre/post. C′ in parallel if approved. |
| 12 | Statistics, figures, writing. Internal draft → supervisor review → ICASSP submission (deadline ~Sept 2026). |

Slack plan: if artifacts are rarer than expected, weeks 7–9 pivot to the "why is audio different" analysis (training-duration/scale/tokenization contrasts) — same schedule, different §6.

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Scooped on "artifacts exist in audio model X" | low-moderate (12-month window, field is hot; the word "register" entered speech venues in 2026 via the streaming-ASR papers in §1.7 — vocabulary is leaking, even if the artifact question is untouched) | Our moat is breadth (6–8 models, census + fix + benchmarks). A single-model finding elsewhere doesn't kill a systematic study. Move fast on Phase A; consider an early arXiv preprint of the census to timestamp priority. |
| No artifacts anywhere | low (AST literally starts from DeiT weights) | Branch (c) of §2: the negative-result paper with mechanism analysis. |
| Frame-based models (no CLS, 1D) need protocol changes | certain | Paper #5 already validated the no-CLS workaround (mean received attention, pooled tokens); adopt it. |
| Test-time register fix works worse in audio | possible | That's a result, not a failure — quantify *why* (register count? sequence length? emergence depth?) and the ablations become the story. |
| Reviewer: "just a transfer of a vision result" | certain to be raised | Lead with the audio-native findings (silence law, time/frequency asymmetry, sequence-length scaling, supervised-vs-SSL contrast), not the port. The fix is the utility; the characterization is the science. |
| Kaggle session limits (12 h) vs long extraction jobs | certain | Shard extraction per model per dataset; checkpoint features to Kaggle datasets between sessions. |

## 8. What to tell your supervisor you need

1. Green light on the topic and the 12-week plan.
2. ~30 A100-hours for Phase C′ (optional but strengthens the paper), ~200 GB storage.
3. Weekly 30-min check-ins; the Week-5 mid-project writeup as the go/no-go review.
4. Their read on venue: ICASSP 2027 (deadline ~Sept 2026) as primary — 4-page format suits the census+fix; TASLP later for the extended version.

## 9. Evidence of the gap (reproducible searches)

arXiv API queries run 2026-07-14/15, **re-run and confirmed 2026-07-16** (repeat before submission to check for new entrants):

```
all:"register tokens" AND cat:cs.SD        → 1 tangential hit (CAV-MAE Sync)          [unchanged 07-16]
all:"registers" AND cat:eess.AS            → nothing relevant except the two
                                             streaming "Online Register" papers (§1.7) [new 07-16]
all:"attention sink" AND cat:eess.AS       → 4 hits; only Llama-AVSR relevant
                                             (decoder, not encoder); others are
                                             XLSR-Transducer (sink-as-tool), vocal
                                             separation, serving-systems              [07-16]
all:"massive activations" AND all:"audio"  → 1 hit (same paper)                       [unchanged]
all:"HuBERT" AND all:"high-norm"           → 0 hits                                   [unchanged]
all:"artifact tokens" + audio cats         → 0 hits                                   [07-16]
all:"test-time registers"                  → 2 hits: Jiang et al. + diffusion-DiT
                                             outlier paper — still zero audio         [07-16]
```

Citation-graph sweep (Semantic Scholar API, 2026-07-16): 912 citers of Darcet et al. → 16 audio/speech-titled, none an encoder-artifact study; 39 citers of Jiang et al. → 0 audio.

Contrast: `ti:"registers" AND cat:cs.CV` returns ~10 papers in the last 4 months alone. The topic is hot; the modality is empty.

**Before the supervisor meeting, optionally re-verify on:** Semantic Scholar (search "audio transformer register tokens", "attention artifacts speech model"), OpenReview (ICASSP/ICLR submissions), and Google Scholar alerts on "register tokens audio". Set a weekly alert — cheap insurance.

## 10. Reading list, in order (≈1 week of evenings)

1. **Darcet et al., ICLR 2024** — the phenomenon. Read fully. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
2. **Jiang et al., NeurIPS 2025 spotlight** — the method you'll port; then clone their code and run their demo. [arXiv:2506.08010](https://arxiv.org/abs/2506.08010)
3. **AST, Interspeech 2021** — your bridge model; §2 (architecture) carefully. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
4. **Reassessment, 2026** — skim for protocol & terminology (§3). [arXiv:2603.25803](https://arxiv.org/abs/2603.25803)
5. **PH-Reg, NeurIPS 2025** — method section only, for the optional baseline. [arXiv:2505.21501](https://arxiv.org/abs/2505.21501)
6. **Llama-AVSR sinks, 2025** — short; your related-work differentiation. [arXiv:2510.22603](https://arxiv.org/abs/2510.22603)
7. Background skims: HuBERT [2106.07447](https://arxiv.org/abs/2106.07447), WavLM [2110.13900](https://arxiv.org/abs/2110.13900), BEATs [2212.09058](https://arxiv.org/abs/2212.09058), Audio-MAE [2207.06405](https://arxiv.org/abs/2207.06405), Whisper [2212.04356](https://arxiv.org/abs/2212.04356), EAT [2401.03497](https://arxiv.org/abs/2401.03497) — architecture/tokenization sections only.
8. Context on sinks in LLMs (for related work): StreamingLLM (attention sinks, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)), "Massive Activations in LLMs" ([arXiv:2402.17762](https://arxiv.org/abs/2402.17762)).
9. Adjacent-work skims from the 2026-07-16 sweep (§1.7 — abstracts + method figures only): Online Registers for streaming S3Ms [2602.23702](https://arxiv.org/abs/2602.23702) / OPC [2606.21268](https://arxiv.org/abs/2606.21268), "Silence is Sweeter Than Speech" [2205.03759](https://arxiv.org/abs/2205.03759), Whisper-decoder outlier quantization [2406.11022](https://arxiv.org/abs/2406.11022), attention-sink survey [2604.10098](https://arxiv.org/abs/2604.10098).

*Audio prerequisites (deliberately minimal):* one afternoon plotting `librosa.display.specshow` on speech/music/silence until you can visually identify silence, harmonics, and broadband noise. You do **not** need Fourier theory, filter design, or vocoders for this project — the models consume mel spectrograms or raw waveforms through frozen frontends, and all our analysis lives in transformer-internals space you already know from ViTs.

---
*All six anchor/reference TeX sources are in `Papers_Tex_Source/` (flattened as `flattened_main.tex` in each folder) and PDFs in `Papers_pdf/`. Numbers quoted from anchor papers in this plan were extracted from those sources directly.*
