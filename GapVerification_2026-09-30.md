# Gap Re-Verification & A* Viability Assessment — 2026-09-30

**Subject:** *Do Audio Transformers Need Registers?* (see `ResearchPlan.md`, dated 2026-07-15)
**Question asked:** is the gap still open, is the idea still doable, and can it reach an A\* venue?
**Verdict in one line:** the gap is **still open but materially narrower**, the plan's **central hypothesis is probably false as written**, both of the plan's primary venues **closed in the last 7 days**, and the project is **A-venue / A\*-workshop shaped as currently framed** — but there is a specific re-scope that makes an A\* main-track submission genuinely plausible.

---

## 1. What was actually checked (reproducible)

All queries run 2026-09-30. Scripts and raw JSON in the session scratchpad; re-runnable.

### 1.1 Citation-graph sweep (Semantic Scholar Graph API, full pagination)

| Anchor | S2 citers today | (plan said, 2026-07-16) |
|---|---|---|
| Darcet et al., *ViTs Need Registers* (2309.16588) | **1,018** | 912 |
| Jiang et al., *ViTs Don't Need Trained Registers* (2506.08010) | **47** | 39 |
| PH-Reg, *Self-Distilled Registers* (2505.21501) | 17 | — |
| *Denoising Vision Transformers* (2401.02957) | 55 | — |
| *When Attention Sink Emerges* (2410.10781) | 221 | — |
| *Massive Activations in LLMs* (2402.17762) | 303 | — |

Across all **1,661 citing papers**, the intersection of {audio/speech vocabulary} × {register / high-norm / attention-sink / massive-activation vocabulary} yields **9 papers**. None characterizes artifacts in a pretrained audio encoder. The 9:

1. `2609.20849` **SPARE** (Aug 2026) — adds *one* register token to a Large Audio **Language** Model, supervised with a Sentence-BERT cosine loss, to improve audio reasoning. Register-as-device, no artifact analysis. Cite.
2. `2603.14337` **On the Nature of Attention Sink … in Omni-LLMs** (Mar 2026) — sinks in Omni-LLM **decoders**. Cite & differentiate.
3. `2605.10815` **Probing Cross-modal Information Hubs in AV-LLMs** (May 2026) — sink tokens as cross-modal hubs, AV**LLM** decoder. Already in plan §1.7.
4. `2602.23702` **Online Register for Dual-Mode S3Ms** (ICASSP 2026) + `2606.21268` **Online Predictive Coding** (Jun 2026) — registers as streaming-ASR placeholders. Already in plan §1.7.
5. `2510.22603` **Llama-AVSR sinks** — plan §1.5. LLM decoder.
6. `2509.20986` **SiNGER** — vision distillation.
7. `2505.01237` **CAV-MAE Sync** — registers as a design choice.
8. `2603.11647` OmniForcing, `2608.00058` ECG — false positives.

**Jiang et al. (the method being ported): 47 citers, zero audio.** PH-Reg: 17 citers, zero audio. This part of the plan holds.

### 1.2 arXiv API sweep (34 keyword × category queries)

| Query | Hits | Assessment |
|---|---|---|
| `"high-norm" AND (cat:cs.SD OR cat:eess.AS)` | **0** | gap holds |
| `"outlier tokens" AND (cat:cs.SD OR cat:eess.AS)` | **0** | gap holds |
| `"attention artifacts" AND (cat:cs.SD OR cat:eess.AS)` | **0** | gap holds |
| `"register neurons"` (any category) | 1 (Jiang only) | method still unported |
| `"test-time registers"` | 3 — Jiang, DiT-outliers, `2607.16824` (image generation) | still zero audio |
| `"register tokens" AND (cs.SD OR eess.AS)` | 2 — SPARE, CAV-MAE Sync | tangential |
| `"massive activations" AND (cs.SD OR eess.AS)` | 1 — Llama-AVSR | LLM decoder |
| `"attention sink(s)" AND (cs.SD OR eess.AS)` | 5 — all KV-cache/serving/separation/streaming engineering | tools, not analysis |
| `ti:"registers" AND cat:cs.CV` | **40+, ~15 new since July** | vision niche still accelerating |

### 1.3 Venue-freshness check

ICLR 2027 submissions are **not yet public** on OpenReview (API returns 403 for `ICLR.cc/2027/Conference/-/Submission`). The deadline was **2026-09-25**. Anything submitted there is invisible for now — this is the single largest residual scoop risk and should be re-checked when the listing opens (typically within 2–4 weeks of the deadline).

---

## 2. The one real hit: a partial scoop on AST

**"Silence in the Noise: Probing Attention Artifacts and Register Tokens in Audio Spectrogram Transformers"**
Suleman Imdad (Johns Hopkins, Whiting School), **Zenodo preprint, 2026-06-16**, DOI `10.5281/zenodo.20710635`, 15 pages, IEEE two-column, code at `github.com/real1900/silence-in-the-noise`.

This did not appear in the plan's July sweep, and it is **invisible to every structured search**: not on arXiv, **not indexed in Semantic Scholar** (title search returns 0), no citations, single author, no venue. It surfaces only via web search. It is, on the evidence of its own artifacts (`Imdad_Final_Research_Paper.ipynb`, "Final Research Paper", vast.ai spot instance, acknowledgment to the School for compute), a graduate course project.

### What it actually establishes

- **One model:** `MIT/ast-finetuned-audioset-10-10-0.4593`. **One dataset:** ESC-50. Diagnostics on **64 clips**.
- Layer-wise max/median patch-token L2 norm across all 12 layers: **1.31–1.93**, final layer **1.23**, outlier rate **0.0%** in the last three layers.
- Trained (Darcet-style) registers, `n ∈ {0,2,4,8,16}`, fine-tuned on ESC-50: registers absorb 11.2% of attention mass at n=16 (~8.5× chance); patch–patch attention Frobenius norm falls ~9% monotonically; accuracy **+0.33 pp, p = 0.12**; cross-fold σ 1.035 → 0.727 pp, **p = 0.15**. Honestly reported as "no harm done with a mild positive trend."
- INT8 PTQ on three backends: cosine fidelity 0.980–0.9988, 49–74% size reduction. Proposes max/median final-layer norm ratio as an INT8-readiness diagnostic. **This is the paper's actual contribution** — the register part is secondary.

### What it does NOT do — i.e. what is still yours

- **No second model.** SSAST, PaSST, BEATs, HTS-AT, AudioMAE, M2D are named in its own future work as unresolved. **No frame-based / speech model at all** — the strings "HuBERT", "WavLM", "Whisper", "wav2vec" appear **zero times** in the paper.
- **No localization analysis.** Despite the title, "silence" appears 7 times, all in prose. Zero energy-vs-outlier correlation, zero spectral flatness, zero neighbour-redundancy measure, zero synthetic silence/SNR diagnostics. **RQ2 is completely untouched.**
- **No information probes.** No position prediction, no patch reconstruction, no single-token classification. **RQ3 untouched.**
- **No `FindRegisterNeurons`, no test-time registers.** It fine-tunes trained registers; it never touches Jiang et al.'s mechanism or training-free edit. **RQ4 untouched.**
- **No dense/frame-level task.** Its own limitations section: *"Classification is the wrong end task… dense audio prediction tasks (sound event detection, audio segmentation) are the natural setting for register tokens, and we expect future work in those settings to find substantially larger and easier-to-detect benefits."* **RQ5 untouched, and pre-endorsed.**

### A methodological flaw you must correct and can cite

Its own Method §IV-D: *"For each token i, we compute ‖h⁽ᴺ⁾ᵢ‖₂ in the final layer **(after AST's terminal LayerNorm)**."*

LayerNorm rescales every token toward a common norm. Measuring the headline diagnostic **after** the terminal LN mechanically compresses the max/median ratio toward 1 whether or not artifacts exist — a final-layer ratio of 1.23 is roughly what post-LN features must give. Darcet et al. and the 2026 reassessment (`2603.25803`) both specify **before** the final LayerNorm for exactly this reason; the reassessment makes it part of its terminology hygiene. Its per-layer sweep (1.31–1.93) is measured on the unnormalised residual stream and so is a real number, but the headline claim rests on the flawed one.

**Net effect on you:** the AST-specific claim is planted but weakly established and undiscoverable. You should cite it, reproduce it correctly (pre-LN, more clips, more datasets), and say what it got right and what the measurement convention cost it. That is a contribution, not a liability. It does **not** close the project. It does, however, kill the framing "nobody has asked this question about audio" — the scoped claim must now be *"no systematic, multi-model, multi-tokenisation study of artifact tokens in pretrained audio and speech encoders exists."*

---

## 3. The harder problem: the plan's central hypothesis is probably false

This matters more than the scoop. Three independent lines of evidence now point the same way.

**(a) Darcet's own scaling condition.** Artifacts appeared in DeiT-III **Large/Huge** and DINOv2 **L/H/g**, and largely *not* in Tiny/Small/**Base**. The plan's model grid is almost entirely ViT-Base-scale: AST 87M, BEATs ~90M, Audio-MAE 86M, EAT 88M, HuBERT-Base 95M, WavLM-Base+ 95M, Whisper-small 88M. The plan's §1.6 argument — *"AST is a ViT, initialised from DeiT weights, so if artifacts exist anywhere in audio, AST is where we'll find them"* — is backwards. AST is initialised from DeiT at the **scale where DeiT does not have artifacts**, and then trained on AudioSet (~5k h), orders of magnitude less than LVD-142M/DINOv2.

**(b) Independent null results in the same niche.** `2605.16147` (pixel-space DiTs) and `2605.05206` (Taming Outlier Tokens in DiTs) both report that DiTs do **not** show high-norm patch outliers. The reassessment `2603.25803` reports that several register claims do not generalise across vision architectures. The phenomenon is narrower than the 2024 framing suggested.

**(c) The Zenodo AST null result** — flawed headline diagnostic, but its pre-LN per-layer sweep still tops out at 1.93×, nowhere near the ~10× that defines the phenomenon.

**Consequence.** The plan's §2 "answer-independence" argument ("all three outcomes are publishable") is true for a *workshop or Interspeech-shaped* paper, and false for an A\* main track. Branch (c) — "Audio Transformers Don't Need Registers" — is an **A\*-workshop paper**. No NeurIPS/ICML/ICLR area chair accepts *"we checked eight Base-scale encoders for a phenomenon the literature says requires Large-scale, and did not find it."* You would be running a 12-week project whose most likely outcome is an A-tier paper.

**The good news:** the null-result risk is concentrated almost entirely in *which models you chose*, and that is free to fix.

---

## 4. Re-scope: where the A\* paper actually is

Two changes. Both are cheap; together they change the paper's category.

### 4.1 Move the grid up the scale axis, and add the models built for it

Drop the Base-only framing. Add, in priority order:

| Model | Enc. depth × width | Pretraining scale | Why it is the right candidate |
|---|---|---|---|
| **Whisper large-v3 encoder** | 32 × 1280 | 1M–5M h weak supervision | Deepest, widest, longest-trained audio encoder in public release. Highest prior. |
| **Whisper medium / small enc.** | 24×1024 / 12×768 | same corpus | **Scale ladder at fixed data and objective** — a clean causal test of the emergence-by-scale claim that vision can only run across differently-trained models. |
| **XLS-R-1B / 2B**, MMS-1B | 48 × 1280 | 436k h, 128 langs | Largest SSL speech encoders. |
| **Dasheng-1.2B** | large | general audio | Largest general-audio encoder; scale regime, non-speech domain. |
| **HuBERT-Large, WavLM-Large** | 24 × 1024 | 60k / 94k h | Plan already has these; they become core, not comparison. |
| **BEATs-Large (~300M)** | ViT-L-ish | AudioSet-2M | Patch-based at L scale — the missing cell in the Zenodo study. |
| **EAT** | 88M | data2vec-style bootstrap | Only Base model worth keeping as a *primary*: self-distillation is the exact regime that produced DINOv2's artifacts. |
| AST, Audio-MAE, HuBERT/WavLM-Base | Base | — | keep as the **negative-control arm** and as the correct reproduction of the Zenodo result |

This alone converts "did we find it?" into "**at what scale does it switch on, holding data and objective fixed?**" — which is a result whether the switch-on point is inside the public model range or above it.

### 4.2 Point it at Whisper's silence hallucination — the hook the plan is missing

This is the part I would build the paper around.

**The known problem.** Whisper large-v3 emits words on **61.9%** of pure room-tone clips and emits *something* on **100%** (`2609.32560`, 2026-09-26 — four days ago); 40.3% non-empty on non-speech inputs is the widely-cited figure. It is a live production problem with real downstream harm, and the literature on it is very active *right now*:

- `2609.32560` **How to Reduce Whisper Hallucination** (Sep 2026) — synthetic-positive data augmentation + fine-tuning.
- `2609.04561` **Hallucination Space Projection** (Sep 2026) — **decoder**-state projection at decode time.
- `2608.15940` **The Null Token Knows** (Aug 2026) — **decoder** null-token logit shifts; explicitly does *not* cite Darcet, does *not* analyse norms/sinks/registers.
- `2607.01108` NPUsper — detects hallucinated tokens from decoder temporal patterns.
- Plus VAD preprocessing and pseudo-label filtering as the standing practice.

**Every published attack is decoder-side or data-side. Nobody has asked what the encoder is doing with the silence.** And the register framework is the natural instrument, because Whisper hands you something vision has never had:

> **Whisper's encoder always consumes a fixed 30-second mel window.** A 3-second clip is padded with ~27 seconds of *exactly zero-information input*, of controllable length and controllable position.

That is a **causal, gradable, ground-truth-known uninformative region**. Darcet's "redundant patch" theory has only ever been tested *correlationally* — you measure neighbour cosine similarity on natural images and report an association. With Whisper padding you can *dial the amount of null input* (1s speech + 29s pad → 29s speech + 1s pad) and watch whether the high-norm token count, their positions, and the number of test-time registers required scale with it. Vision cannot run this experiment. **This is the mechanistic contribution that makes the paper A\*-shaped rather than a modality census.**

And it closes into a practical result: if the encoder's non-speech regions carry high-norm register-like tokens that the decoder's cross-attention then reads, you have (i) a *mechanistic explanation* for silence hallucination that the entire 2026 hallucination literature is missing, and (ii) a **training-free** fix in the form of a test-time register that relocates that computation out of the padded frames — competing against methods that all require fine-tuning or decoder surgery.

**Recast contributions:**

- **C1** First multi-family, multi-tokenisation, **multi-scale** census of high-norm artifact tokens in pretrained audio and speech encoders (12–14 checkpoints, patch vs frame, supervised vs SSL vs weak supervision, Base→1B), with the correct pre-LayerNorm protocol. Includes the corrected reproduction of the only prior AST measurement.
- **C2** **The padding experiment.** Causal, gradable test of the token-recycling/redundancy account using Whisper's fixed 30 s window — the first non-correlational test of that theory in any modality.
- **C3** Mechanistic account of **Whisper silence hallucination** as an encoder-side artifact phenomenon, plus a **training-free** test-time-register intervention, benchmarked against the 2026 decoder-side and data-side baselines on their own benchmarks.
- **C4** Frame-level downstream suite (SED / PSDS, attention-as-localisation) where the vision literature predicts the gains are, and which the Zenodo paper's own limitations section flags as the right setting.

Note that under this framing the negative branch is no longer fatal: "the switch-on scale for artifacts is above the public audio-encoder range, and here is the padding experiment showing that null input alone is insufficient without scale" is a genuine mechanistic result about *when* the phenomenon emerges, testable only in audio, and it lands at an A\* workshop or Interspeech comfortably — while the positive branch on Whisper-large is an A\* main-track paper.

---

## 5. Venues — the plan's schedule is broken

| Venue | Tier | Deadline | Status |
|---|---|---|---|
| **ICASSP 2027** | CORE B | **2026-09-23** | **PASSED 7 days ago.** The plan's primary venue is gone for this cycle. |
| **ICLR 2027** | **A\*** | **2026-09-25** | **PASSED 5 days ago.** |
| **ICML 2027** | **A\*** | abstract **2027-01-16**, paper **2027-01-22** | **Primary A\* target. ~16 weeks from today** — fits the 12-week plan plus a writing buffer, with no slack. |
| Interspeech 2027 | CORE A | 2027-02-09 | Realistic floor; 3 weeks after ICML, so an ICML rejection recycles immediately. |
| **ACM MM 2027** | **A\*** | TBA (historically ~April) | Strong A\* fallback; ACM MM takes audio work seriously. |
| **NeurIPS 2027** | **A\*** | ~May 2027 | Fallback with room for the full journal-scale version. |
| ICLR 2027 **workshops** | A\* workshop | ~Feb 2027 (papers) | The "at least an A\* workshop" floor the brief asks for. |
| TASLP | Q1 journal | rolling | Extended version. |

Two notes. First, the plan's §6 week-12 milestone ("ICASSP submission, deadline ~Sept 2026") is now unachievable — the schedule must be rewritten against **2027-01-22**. Second, ICASSP is **CORE B**, and Interspeech is **A**, not A\*: the original venue list never targeted A\* at all. That mismatch with the stated ambition needs to be settled with the supervisor explicitly, because it changes what the paper has to contain, not just when it is due.

---

## 6. Go/no-go: one experiment, ~2 days, before anything else is built

Everything above hinges on one measurement. Do it first, in week 1, before touching probes, benchmarks, or the PH-Reg baseline.

**Protocol.** For each checkpoint in {Whisper small/medium/large-v3 enc., XLS-R-300m/1B, WavLM-Large, HuBERT-Large, BEATs-Large, Dasheng, EAT, AST, Audio-MAE}: hook the **residual stream at every block output, before the terminal LayerNorm**; 500 clips spanning LibriSpeech test-clean, ESC-50, FSD50K eval, and a synthetic set of {1,3,5,10,20,29}-second clips padded to 30 s. Record per-token L2 norm, max/median and 98th-percentile ratios per layer, Hartigan dip test for bimodality, mean received attention at outlier positions, and outlier position vs frame log-energy / padding mask.

**Decision rule — pre-register it.**

- **Any model shows max/median ≥ 5× with a bimodal norm distribution and outliers concentrated on padded or low-energy frames** → green light on the full C1–C4 paper, ICML 2027. This is the strong outcome and Whisper large-v3 is where to expect it.
- **Ratios in the 2–5× band, or positive only in the largest models** → the scale-ladder result (C1 + C2) is the paper; the hallucination link becomes a section rather than the headline. Interspeech 2027 / A\* workshop, with ACM MM 2027 as the stretch.
- **Everything under 2× including Whisper large-v3 and XLS-R-1B** → do **not** run the 12-week plan. Pivot to the negative-result mechanism paper (C2 padding experiment + the emergence-condition analysis: softmax normalisation, pre- vs post-LN, bidirectional vs causal masking, training compute), target an A\* workshop, and reclaim the remaining 8 weeks.

Cost: a few T4-hours. It is worth strictly more than the rest of week 1 combined, and the plan currently schedules it as part of a broad week 2–3 census. Pull it forward.

---

## 7. Related work to add before a reviewer finds it

Beyond plan §1.7, all verified today:

**Vision/theory (the niche has moved a lot since July):**
- `2602.22394` **Vision Transformers Need More Than Registers** — attributes artifacts to "lazy aggregation": ViTs use background patches as shortcuts for global semantics. Directly relevant competing mechanism; 12 benchmarks.
- `2605.19622` **UniRefiner** — argues the "high-norm token" definition is too narrow.
- `2607.16824` **Test-Time Registers as Global Priors for Tokenized Image Generation** (Jul 2026) — the method you are porting, ported elsewhere first.
- `2608.10989` Task Registers for token pruning; `2606.14701` RATS; `2606.12036` face recognition; `2510.04547` **Activation Quantization of Vision Encoders Needs Prefixing Registers** (registers × quantization — overlaps the Zenodo paper's angle).
- `2604.14433` Zero-ablation overstates register content dependence (already local).
- `2502.07004` **Demystifying Singular Defects in LLMs** (ICML 2025) — high-norm tokens modelled via singular vectors of layer linear approximations. Gives you an analytical prediction to test in audio encoders.
- `2603.17771` **Attention Sinks Induce Gradient Sinks** (Mar 2026) — massive activations as RMSNorm-mediated gradient regulators, **under causal masking**. Important for you: audio encoders are *bidirectional*, so this mechanism should not apply — a sharp, testable prediction.
- `2604.10098` attention-sink survey — confirmed to have **no audio-encoder category**.
- `2405.14858` Mamba-R — relevant if you add audio-Mamba/SSM encoders.

**Audio-side (must cite and differentiate):**
- **`zenodo.20710635`** Imdad, *Silence in the Noise* — §2 above. Cite it.
- `2504.01690` / IEEE OJSP 2025, Lee et al., **Token Pruning in Audio Transformers** — **the closest prior work to your RQ2.** Reports high correlation between AST/AudioMAE attention scores and patch intensity/variation, finds low-intensity tokens remain important, and that AudioMAE retains more low-intensity tokens than AST (attributed to the reconstruction objective). Your RQ2 AUROC analysis overlaps this; differentiate on norms-vs-attention and on causal intervention.
- `2507.02666` **ASDA** — differential attention for audio SSL, motivated explicitly by attention "allocated to irrelevant information". SOTA on AS-2M/ESC-50/SPC-2. A competing *architectural* fix for what may be the same phenomenon; no norm analysis. Must be discussed.
- `2406.11022` Whisper PTQ with gated attention — confirms outliers exist in Whisper weights **and activation tensors**; encoder/decoder split not resolved from the abstract. **Evidence in your favour**, and the paper to differentiate from on token-level-vs-tensor-level.
- `2504.14915` **StableQuant** — layer-adaptive PTQ for HuBERT/wav2vec2 because of per-layer scale distributions. Same category of indirect evidence.
- `2609.32560`, `2609.04561`, `2608.15940`, `2607.01108` — the 2026 Whisper-hallucination baselines your C3 must beat or at least be positioned against.
- `2607.00387` *From Objectives to Applications: Aligning Architectural Biases in Audio SSL* (Jul 2026) — useful framing citation for the objective×architecture contrasts in your grid.
- `2205.03759` *Silence is Sweeter Than Speech* — already in the plan; now load-bearing for C2/C3, not just motivation.

**Wording fix.** Plan §2 says "No paper analyzes high-norm artifact tokens / register tokens in *any* pretrained audio encoder." That is now false. Replace with: *"The only prior measurement is a single-checkpoint, single-dataset preprint on AST that measures norms after the terminal LayerNorm; no multi-model, multi-tokenisation, multi-scale study exists, and no audio work has ported the register-neuron mechanism or the training-free test-time fix."*

---

## 8. Bottom line

| Question | Answer |
|---|---|
| Is the gap still there? | **Yes**, for the project as a whole. 1,661 citing papers, 34 arXiv queries: zero systematic audio-encoder artifact studies, zero audio ports of `FindRegisterNeurons`/test-time registers, zero frame-based-model analyses. |
| Did anyone fill it? | **Partially, for AST only** — one undiscoverable Zenodo course-project preprint with a flawed headline diagnostic. RQ2/RQ3/RQ4/RQ5 are all untouched, and it pre-endorses your RQ5. |
| Still doable on the stated compute? | **Yes**, with the grid change. Whisper-large-v3 encoder and XLS-R-1B are fp16-inference feasible on a T4; the whole census stays inference-only. Note the local RTX 4050 has 6 GB — Kaggle remains the right target. |
| Good enough for A\*? | **Not as framed.** A multi-model census plus a method port is an Interspeech/ICASSP/TASLP paper. With the scale ladder, the Whisper padding experiment, and the silence-hallucination mechanism, it is a credible ICML 2027 submission. |
| A\* workshop floor? | **Comfortably yes**, on every branch including the fully negative one. |
| What is the single biggest risk? | **Not** being scooped. It is finding nothing because the chosen models sit below the emergence scale. Run §6 first. |
| Timeline | ICASSP 2027 and ICLR 2027 both closed in the last week. Rewrite the schedule against **ICML 2027, 2027-01-22** (~16 weeks), Interspeech 2027 2027-02-09 as the immediate recycle, ACM MM 2027 / NeurIPS 2027 beyond. |

**Re-check before submission:** ICLR 2027 OpenReview listing when it opens (largest residual scoop risk), plus a repeat of the §1 sweeps and a plain web search for the Zenodo paper's author and title — structured indices did not find it, so structured indices will not find the next one either.
