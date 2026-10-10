# Scoop & novelty check — 2026-10-10

Question checked: has anyone (a) characterised register / high-norm / massive-activation / attention-sink tokens **inside speech or audio encoders** (Whisper, HuBERT/WavLM/wav2vec2, AST/BEATs/EAT, LALM audio towers), (b) linked them to **silence or Whisper's 30-s padding**, or (c) published a **training-free padding-free Whisper** method based on cached silence?

## Sources scanned (all on 2026-10-10)

| Source | Scope | Method | Relevant hits |
|---|---|---|---|
| OpenReview — ICLR 2027 submissions | all 42,357 submissions (849 with audio terms in title/keywords) | API pull through the user's browser session; regex over title/abstract/keywords | 74 sink/register/massive-activation submissions, **none** on audio encoders. Audio-adjacent only: *Why Full-Duplex Speech Models Collapse in Long Conversations* (first-KV "anchor" eviction in duplex LMs), *Audio Token Attention Is Predictable…* (Triage, LALM token pruning), LALM interpretability (*How Sound Meets Language*, *Selective Listening*, *Can We Read the Mind of an Audio LLM?*), LALM hallucination (*From Attention Weights to Audio Writes*, *From Attention to Semantics*). |
| OpenReview — ICML 2026 accepted | all 6,341 papers | same | 24 sink/massive/register papers; audio: only *Probing Cross-modal Information Hubs in AV-LLMs* (LLM-side sinks). |
| Papers-with-Code conference tags | ICLR 2026, ICML 2026, CVPR 2026, ECCV 2026, ACL 2026, AAAI 2026, COLM 2026 | `pwc paper list --conference … --search …` | vision/LLM register & sink papers only (e.g. CVPR'26 *ViTs Need More Than Registers*, ECCV'26 *RegCache*, ICLR'26 *To Sink or Not to Sink*, ICML'26 *Anatomy…*, *Structural Origin…*, *Single Layer…*). No audio-encoder work. |
| ISCA archive — Interspeech 2026 proceedings | full index | keyword scan + abstract reads | none on encoder registers/sinks. Nearby: *Grounding Whisper* (anchor audio for hallucination detection), *Not All Frames Are Equal* (DiffAQ: zero-padded frames dominate PTQ calibration), *Evolution-Strategy Calibration* (large audio activation ranges), *Systematic PTQ … Whisper*, Whisper hallucination detection/steering papers. |
| arXiv API | 2023 → 2026-10-10 | boolean field queries (abs/ti/co) incl. `whisper AND (attention sink OR high-norm OR massive activation OR register)`, `(hubert OR wavlm OR wav2vec) AND (sink OR high-norm …)`, `speech AND test-time registers`, `TTS/audio generation AND massive activations` | none. Only "registers" in the unrelated sense (online registers for streaming SSL, 2602.23702; global Whisper token for retrieval, 2601.15118). |
| Semantic Scholar (bulk + relevance) | 2025–2026 | boolean queries; citation-graph sweep of 16 anchor papers (session 1: ~4,200 citers) | none beyond LLM-side AVSR/Omni sink papers (2510.22603 ICASSP'26; 2603.14337; 2605.10815 ICML'26). |
| NeurIPS 2026 accepted | OpenReview venue not yet public (count 0) | arXiv comment search `co:"NeurIPS 2026"` | no audio-encoder sink/register paper found. |
| Web / GitHub (padding-free Whisper) | — | web search | WhisperFlow hush word (trained), WhisperKit silence caching (block masks + self-distillation), FUTO ACFT checkpoints (fine-tuned short audio context, ~8× encoder speed-up), NPUsper (online hallucination detection), ICASSP'26 *Adapting Whisper for Padding-Free Inference* (fine-tuning + KD), Stride-k (token dropping). **No training-free cached-silence method found.** |

## Verdict
- (a) and (b): **open** as of 2026-10-10 across ICLR 2027 submissions, ICML 2026, Interspeech 2026, CVPR/ECCV/ACL 2026 and arXiv. A non-peer-reviewed Zenodo note (Imdad, JHU, 2026) reports AST is artifact-free (our pilot agrees and explains why: CLS+DIST absorb up to 45% of attention).
- (c): **open but thin moat** — the trick is simple; it must be presented as a consequence of the mechanism, with broad evaluation.

## Watch list (check weekly; see plan.md §10)
- KAIST MM / J.S. Chung group (Omni-LLM sinks 2603.14337; cross-modal hubs 2605.10815, ICML'26).
- Imperial/Meta Llama-AVSR group (2510.22603, ICASSP'26).
- Triage authors (2609.38878; ICLR'27 submission).
- POSTECH RegCache group (2510.04547, ECCV'26) — natural next step is audio encoders.
- Berkeley test-time-register group (2506.08010).
- Imdad (JHU) Zenodo AST note — could extend to Whisper.
- Whisper efficiency/hallucination systems groups (WhisperFlow/UVA; NPUsper; Argmax/WhisperKit; FUTO).
