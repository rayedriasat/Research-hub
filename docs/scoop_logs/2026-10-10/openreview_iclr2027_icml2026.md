# OpenReview scans (2026-10-10)

Pulled through the OpenReview API v2 in the user's browser session (anonymous API calls are behind a bot challenge).
Regexes were applied to title, abstract and keywords. IDs are OpenReview forum IDs.

## ICLR 2027 — all submissions
- Venue id `ICLR.cc/2027/Conference/Submission`: **42,357** submissions; **849** with audio terms in title/keywords.
- Audio term AND phenomenon term (attention sink / massive activation / high-norm / register token / outlier token / super neuron) anywhere: 6 hits, none relevant except
  - `SbI8ap4e6o` *Why Full-Duplex Speech Models Collapse in Long Conversations* — sliding-window duplex LMs fail when the first KV entry ("initialization anchor") is evicted; fix: pin it.
  - `BXUnpS9Y2b` *Quantization-Conditioned Backdoors on LLMs via Attention Sinks* (not audio).
- Closest audio submissions (no encoder-register / silence-sink content):
  `1DvvJkCmz2` Audio Token Attention Is Predictable Before the Language Model Runs ·
  `GlbZrgNGVP` How Sound Meets Language: Causal Mechanisms of Audio-Language Integration in LALMs ·
  `xi787ClkYW` Selective Listening: Mechanism-Guided Control of Audio Influence in LALMs ·
  `shpQH24STD` Can We Read the Mind of an Audio LLM? ·
  `hvtPM5KpMf` From Attention Weights to Audio Writes (LALM hallucination) ·
  `z122DbXfR2` From Attention to Semantics (sound-event hallucination) ·
  `zXocoL0tlq` Speech recognizers carry the surrounding speech rate but use it less as they grow (Whisper interpretability) ·
  `DO5nEXoZDZ` Mixture Probing in Frozen Audio Encoders ·
  `WAUtbeiFoQ` ECHO: Embedding Convergence in Audio Models ·
  `UREguflpCe` TimePrune · `ef0ZsdpRW4` AudioTETRIS · `ZTw9SWiZ4d` X-AuT (LALM token/encoder compression) ·
  `v5Z8GhEgnv` Qwen2-Audio quantization case study · `BoOIVM70IL` MorphoQuant (omni PTQ outliers).
- Whisper mentioned anywhere in abstracts: 18 submissions; none about padding, sinks, registers or silence.

### Sink / massive-activation / register submissions (title or keywords): 74
Non-audio. Most relevant to our theory tests:
`gptDmR0wMm` What Artifact Tokens in Vision Transformers Are, and What They Do ·
`k8FG3HZdQa` A Unifying View of Attention Sinks: From Mechanisms to Architectural Interventions (ViTs, gating, registers) ·
`EVFODHJYWe` SAGA: Spatially-Aware Gated Attention for Separating Outlier Structure from Functional Effect in ViTs ·
`uCBbLTjlcQ` Functional Specialization and Self-Persistent Routing of Native Tokens in ViTs ·
`TeJQpaiiuJ` Decoding Register Tokens Preserves Global Scene Details ·
`gHOiEtCMQy` It's All About One Direction (DiT massive activations / high-norm) ·
`WKk6vvksuz` The Life Cycle of a Massive Activation: Stochastic Birth, Weight-Decay-Driven Growth, and Competitive Consolidation ·
`ETM5cVneOg` Where Massive Activations Come From: Causal Evidence from MLP Geometry ·
`SC6DFc07Lt` Before Attention Sinks Form: Hidden-State Geometry ·
`lsAebgJ9sd` Few Neurons, Many Roles: Sparse MLP Neurons That Control Attention Sinks ·
`My2EDuC1YH` Attention Sinks in VLMs Are Borrowed, Not Seen ·
`TPbu1cHiCa` Mind the Spike (= arXiv 2609.32808) ·
`e4QlBA8rQz` Which the Eye Fears (= arXiv 2609.35630) ·
`iGn1dhuaQE` Attention Sinks Beyond the First Position: Heterogeneous Pathways ·
`WdMtlG7LnC` Attention Sinks in Transformer-based LMs: Emergence Mechanisms and Benign Effects ·
`hoxR1gpwWP` Attention Sink Functionality Is Query-Conditional ·
`arzW42n0jT` What Makes Position Zero Special? ·
`D4LqhTwVzh` Registers Matter for Pixel-Space DiTs · `OxsMU47r72` Taming Outlier Tokens in DiTs ·
`p0Bgmabr6m` Steering Video DiTs with Massive Activations · `IKAzeN8nlO` Massive Activations in Diffusion MLLMs ·
`8K3gt4rfjA` Massive Activations in Hybrid Linear-Attention LLMs · `OQ7Retjeni` Massive Activation Gating Channel in LLMs ·
`0zbHqHIlZW` When Is an Attention Sink Robust? A Convex-Geometric Theory · `wEhtAx4f1v` Reverse Engineering Attention Sinks in LLMs.

## ICML 2026 — accepted papers
- Venue id `ICML.cc/2026/Conference`: **6,341** accepted papers; 96 with audio terms in title/keywords.
- Sink / massive / register papers: 24, e.g. *The Structural Origin of Attention Sink* (dAWFE1gPmJ), *Anatomy of Massive Activations and Attention Sinks* (VUiMZBfHBe), *A Single Layer to Explain Them All* (qAqp0k2SBP), *Towards Understanding Massive Activations in Attention Sink Mechanism* (rvCehcts1v), *Attention Sinks in Diffusion Transformers: A Causal Analysis* (QwE8cOtclR), *Attention Sinks as Internal Signals for Hallucination Detection* (3NdFFpcEjU), *The Extra Tokens Matter* (lj1Qi7Fgyb).
- The only audio one: *Probing Cross-modal Information Hubs in Audio-Visual LLMs* (nxKRwB1J63) — LLM-side sinks, not encoders.

## NeurIPS 2026
- `NeurIPS.cc/2026/Conference` returned 0 public notes on 2026-10-10 (accepted list not yet public).
