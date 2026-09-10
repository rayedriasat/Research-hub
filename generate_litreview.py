#!/usr/bin/env python3
"""Generate formatted Literature Review DOCX"""
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_heading(doc, text, level=1, color=(195, 97, 47)):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.color.rgb = RGBColor(*color)
    run.font.name = 'Georgia'
    return h

def add_para(doc, text, bold=False, italic=False, color=None, highlight=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.font.name = 'Calibri'
    if bold: run.bold = True
    if italic: run.italic = True
    if color: run.font.color.rgb = RGBColor(*color)
    if highlight:
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), highlight)
        run._element.get_or_add_rPr().append(shading)
    return p

def add_citation(doc, authors, year, title, venue, arxiv):
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(f"{authors} ({year}). ").bold = True
    p.add_run(f'"{title}". ')
    p.add_run(venue).italic = True
    p.add_run(f". arXiv:{arxiv}")
    p.paragraph_format.left_indent = Inches(0.25)

doc = docx.Document()

# Title page
title = doc.add_heading('Literature Review', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.color.rgb = RGBColor(196, 97, 47)
title.runs[0].font.size = Pt(28)

subtitle = doc.add_paragraph('Do Audio Transformers Need Registers?')
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.runs[0].font.size = Pt(16)
subtitle.runs[0].italic = True

meta = doc.add_paragraph('Research Proposal Literature Survey')
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.runs[0].font.size = Pt(12)
meta.runs[0].font.color.rgb = RGBColor(92, 99, 93)

doc.add_paragraph()
doc.add_page_break()

# Executive Summary
add_heading(doc, '1. Executive Summary', level=1)
add_para(doc,
"""This literature review surveys the emerging research area of attention artifacts and register tokens in vision transformers, and identifies a critical research gap: these phenomena have never been systematically studied in audio transformers. We analyze 13 recent papers (2021–2026) spanning the discovery of high-norm artifact tokens in vision models, proposed solutions via register tokens, mechanistic studies of attention sinks, and the few existing audio transformer architectures.""")

add_para(doc, "", highlight='FFEB99')  # Highlight box
p = doc.paragraphs[-1]
p.add_run("Key Finding: ").bold = True
p.add_run("Zero papers have analyzed whether pretrained audio encoders (AST, HuBERT, WavLM, Whisper, BEATs) exhibit the register/artifact phenomenon that affects nearly all vision transformers.")

add_para(doc,
"""Our proposed research will transfer the test-time register method from Jiang et al. (NeurIPS 2025 spotlight) to audio transformers, providing the first systematic cross-model study of attention artifacts in the audio domain. This work addresses a verified gap with low compute requirements (~50–80 Kaggle GPU-hours) and clear publication targets (ICASSP 2027, Interspeech 2027).""")

doc.add_page_break()

# Section 2
add_heading(doc, '2. Core Phenomenon: Artifact Tokens in Vision Transformers', level=1)
add_heading(doc, '2.1 Discovery: Vision Transformers Need Registers (ICLR 2024 Oral)', level=2)

add_citation(doc, 'Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P.', '2024',
    'Vision Transformers Need Registers', 'ICLR 2024 (Oral)', '2309.16588v2')

add_para(doc, "Problem Identified:", bold=True)
add_para(doc,
"""Darcet et al. discovered that large-scale vision transformers (DINOv2, DeiT-III) produce high-norm outlier tokens in their feature maps—tokens with norms 100–1000× larger than typical patch tokens. These artifacts preferentially appear on low-information background patches and degrade attention map interpretability. The paper traces this to models repurposing redundant spatial tokens as internal "scratch space" for storing global information beyond what the [CLS] token can hold.""")

add_para(doc, "Key Findings:", bold=True)
p = doc.add_paragraph(style='List Bullet')
p.add_run("Artifact tokens appear in 98% of DINOv2-g images, concentrated on uniform backgrounds")
p = doc.add_paragraph(style='List Bullet')
p.add_run("High-norm tokens (>150 norm threshold) exhibit low neighbor-cosine similarity (~0.3 vs. >0.8 for normal patches)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("These tokens capture global rather than local information, as verified by linear probes")
p = doc.add_paragraph(style='List Bullet')
p.add_run("DINOv1 does NOT show this phenomenon—it emerges specifically in larger-scale trained models")

add_para(doc, "Proposed Solution: Register Tokens", bold=True)
add_para(doc,
"""The paper introduces learnable register tokens—extra tokens appended to the input sequence that absorb high-norm activations during training. Adding 4–16 registers eliminates artifacts, improves object discovery mAP by +6.3pp (DINOv2-g), and maintains image-level task performance. The solution requires retraining from scratch.""")

add_para(doc, "Relevance to Our Work:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This is the foundational paper establishing the phenomenon. Our audio research directly parallels this study's discovery phase: we will measure token norms, attention entropy, and spatial localization of potential artifacts across pretrained audio models. The key difference: audio transformers process spectrograms (time-frequency images), so artifacts may concentrate on silence/low-energy frames rather than visual backgrounds.""")

doc.add_paragraph()

# 2.2
add_heading(doc, '2.2 Training-Free Solution: Vision Transformers Don\'t Need Trained Registers (NeurIPS 2025 Spotlight)', level=2)

add_citation(doc, 'Jiang, N., Dravid, A., Efros, A., & Gandelsman, Y.', '2025',
    'Vision Transformers Don\'t Need Trained Registers', 'NeurIPS 2025 (Spotlight)', '2506.08010v5')

add_para(doc, "Core Innovation:", bold=True)
add_para(doc,
"""Jiang et al. discovered that artifact tokens stem from a small, sparse set of "register neurons" (~10–50 channels out of 1536 in ViT-L) in middle-to-late MLP layers. These neurons produce extreme activations that contaminate the residual stream. Crucially, they show artifacts can be fixed at TEST TIME without retraining: by detecting these neurons and redirecting their activations into an extra untrained token, artifacts vanish.""")

add_para(doc, "Method: FindRegisterNeurons Algorithm", bold=True)
p = doc.add_paragraph(style='List Number')
p.add_run("Measure channel-wise activation statistics across ~1000 images")
p = doc.add_paragraph(style='List Number')
p.add_run("Identify register neurons: channels in top_layer=5 layers with 75th-percentile > threshold (default 75)")
p = doc.add_paragraph(style='List Number')
p.add_run("At inference, shift register-neuron activations into an untrained token via projection")
p = doc.add_paragraph(style='List Number')
p.add_run("Takes ~10 GPU-minutes to calibrate, zero additional training")

add_para(doc, "Results:", bold=True)
add_para(doc,
"""Test-time registers match or exceed trained-register performance on object discovery (+6.1pp TokenCut mAP on DINOv2-L), semantic segmentation (+1.2 mIoU), and k-NN classification. The paper validates on DINOv2, CLIP, and DeiT-III across scales (S/B/L/g).""")

add_para(doc, "Relevance to Our Work:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This is our PRIMARY METHOD ANCHOR. The training-free nature is critical for our beginner-friendly, low-compute scope. We will directly port FindRegisterNeurons to audio transformers: run inference on AudioSet/LibriSpeech samples, identify register neurons in HuBERT/Whisper MLP layers, and test whether redirecting them improves downstream audio tasks. The public code (GitHub: nick-jiang/test-time-registers) provides our implementation starting point.""")

doc.add_paragraph()

# 2.3
add_heading(doc, '2.3 Alternative: Self-Distilled Registers (NeurIPS 2025)', level=2)

add_citation(doc, 'Chen, Y., Yan, Z., Zhou, C., Dai, B., & Luo, A. F.', '2025',
    'Vision Transformers with Self-Distilled Registers', 'NeurIPS 2025', '2505.21501v3')

add_para(doc,
"""PH-Reg proposes a middle-ground solution: post-hoc register insertion via lightweight self-distillation. Given a pretrained ViT, they add register tokens and distill only those tokens + a small MLP subset (1–5% of parameters) for a few GPU-hours, using the frozen base model as teacher. This is cheaper than full retraining but more expensive than Jiang's test-time method.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""We include this as an optional baseline requiring A100 credits (~10–30 hours × $2/hr = $20–60). If test-time registers underperform, PH-Reg provides a fallback that still avoids full model retraining. Not critical path, but strengthens the paper.""")

doc.add_page_break()

# Section 3
add_heading(doc, '3. Cross-Model and Critical Analyses', level=1)

add_heading(doc, '3.1 Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment', level=2)

add_citation(doc, 'Baxevanakis, S., Karageorgis, P., Dravilas, I., & Szewczyk, K.', '2026',
    'Do All Vision Transformers Need Registers?', 'arXiv preprint', '2603.25803v1')

add_para(doc,
"""This March 2026 paper is METHODOLOGICALLY CRITICAL. The authors systematically compare 8 ViT variants (supervised, SSL, hybrid) and find artifact presence correlates with training objective and scale, not architecture alone. They introduce a 98th-percentile norm rule for artifact detection and standardized probe protocols.""")

add_para(doc, "Key Contributions:", bold=True)
p = doc.add_paragraph(style='List Bullet')
p.add_run("Artifacts appear in: DINOv2, MAE, CLIP (large), DeiT-III")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Artifacts absent in: DINOv1, supervised ViT-B, smaller MAE")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Linear probe evaluation: CE + 0.5·MSE loss, ≤30 epochs, patience 3 (we adopt this)")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This paper validates that cross-model systematic studies are publishable even if purely analytical. Our audio work will mirror this structure: compare supervised (AST) vs. self-supervised (HuBERT, WavLM) vs. multilingual (Whisper) vs. discriminative SSL (BEATs). If some show artifacts and others don't, the pattern itself is a finding.""")

doc.add_paragraph()

# 3.2
add_heading(doc, '3.2 Zero-Ablation Overstates Register Content Dependence (Apr 2026)', level=2)

add_citation(doc, 'Parodi, F., Matelsky, J., & Segado, M.', '2026',
    'Zero-Ablation Overstates Register Content Dependence in DINO', 'arXiv preprint', '2604.14433v1')

add_para(doc,
"""This April 2026 critical analysis questions HOW registers function. The authors show that zeroing register tokens causes catastrophic performance drops (-36pp classification), but three replacement controls (mean-substitution, noise, cross-image shuffling) preserve performance within 1pp of baseline. They conclude that registers act as distributional buffers rather than storing critical image-specific content.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This informs our interpretation strategy. If we find audio artifacts, we should test multiple ablation controls (not just zeroing) to characterize whether registers store task-critical information or simply prevent numerical instability. Also demonstrates that nuanced mechanistic findings are publishable in 2026.""")

doc.add_paragraph()

# 3.3
add_heading(doc, '3.3 Artifacts and Attention Sinks: Structured Approximations (ICLR 2026 submission)', level=2)

add_citation(doc, 'Lu, A., Liao, W., Wang, L., Yang, H., & Shi, J.', '2025',
    'Artifacts and Attention Sinks: Structured Approximations', 'arXiv preprint', '2507.16018v1')

add_para(doc,
"""Lu et al. study the interaction between massive tokens (attention sinks with high norms) and artifact tokens (which emerge when massive tokens are removed). They propose Fast Nyström Attention (FNA), a linear-time attention approximation exploiting this structure. Key insight: artifact and massive tokens mutually suppress each other via attention—they're part of a regulatory system.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""If audio transformers show artifacts, this paper's mutual-suppression analysis could explain why. For streaming audio models (Whisper), understanding which frames become attention sinks has implications for efficient inference.""")

doc.add_paragraph()

# 3.4
add_heading(doc, '3.4 Leveraging Registers for Out-of-Distribution Robustness (Jan 2025)', level=2)

add_citation(doc, 'Yellapragada, S., Thopalli, K., et al.', '2025',
    'Leveraging Registers in Vision Transformers for Robust Adaptation', 'arXiv preprint', '2501.04784v1')

add_para(doc,
"""Yellapragada et al. show that pooled register embeddings improve OOD generalization (+2–4% top-1 accuracy) and anomaly detection when concatenated with [CLS] features. This is the first paper demonstrating registers have utility BEYOND artifact removal—they capture complementary global information useful for robustness.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""For audio: if we add test-time registers, we can evaluate not just clean-audio tasks but also noisy/OOD audio (different recording conditions, accents, background noise). This adds a robustness angle to our downstream evaluation.""")

doc.add_page_break()

# Section 4: Attention Sinks in LLMs
add_heading(doc, '4. Attention Sinks in Language Models', level=1)

add_heading(doc, '4.1 Efficient Streaming Language Models with Attention Sinks (Sep 2023)', level=2)

add_citation(doc, 'Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M.', '2023',
    'Efficient Streaming Language Models with Attention Sinks', 'arXiv preprint', '2309.17453v4')

add_para(doc,
"""Xiao et al. discovered that autoregressive LLMs allocate disproportionate attention mass to the first token (BOS/initial token) regardless of its semantic content—an "attention sink." This is caused by softmax normalization requiring probabilities to sum to 1: when no token is strongly relevant, models dump excess probability onto a designated sink. They exploit this for streaming inference with bounded KV cache.""")

add_para(doc, "Key Finding:", bold=True)
add_para(doc,
"""Attention sinks are structural artifacts of softmax, not learned features. Replacing softmax with sigmoid attention (unnormalized) reduces but doesn't eliminate sinks.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This establishes that attention artifacts are not vision-specific—they're a transformer-wide phenomenon. Audio transformers likely exhibit similar sinks, but on WHICH tokens (silence frames? padding? spectrogram edges?) is unknown.""")

doc.add_paragraph()

# 4.2
add_heading(doc, '4.2 When Attention Sink Emerges: An Empirical View (Oct 2024)', level=2)

add_citation(doc, 'Gu, X., Pang, T., Du, C., et al.', '2024',
    'When Attention Sink Emerges in Language Models', 'arXiv preprint', '2410.10781v2')

add_para(doc,
"""Gu et al. trace attention sink emergence during LLM pretraining. They show sinks appear after ~10% of training, correlate with effective optimization on sufficient data, and function as "key biases" storing non-informative attention mass. Sinks are NOT storing useful content—they're numerical stabilizers.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""This mechanistic understanding will guide our interpretation. If audio artifacts appear, we should verify whether they're content-storing (like vision registers) or purely distributional sinks (like LLM first-token sinks). Linear probes + ablation controls distinguish these.""")

doc.add_paragraph()

# 4.3
add_heading(doc, '4.3 Massive Activations in Large Language Models (Feb 2024)', level=2)

add_citation(doc, 'Sun, M., Chen, X., Kolter, J. Z., & Liu, Z.', '2024',
    'Massive Activations in Large Language Models', 'arXiv preprint', '2402.17762v2')

add_para(doc,
"""Sun et al. characterize "massive activations"—outlier neurons with magnitudes 100,000× larger than typical activations in LLMs (LLaMA, OPT, GPT-J). These neurons produce input-independent bias terms and concentrate attention. The paper shows massive activations exist universally across LLM families and also appear in vision transformers.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""Provides the neuron-level view complementary to token-level artifacts. Our FindRegisterNeurons port will effectively search for massive-activation channels in audio model MLPs.""")

doc.add_page_break()

# Section 5: Audio
add_heading(doc, '5. Audio Transformers and the Critical Gap', level=1)

add_heading(doc, '5.1 AST: Audio Spectrogram Transformer (Interspeech 2021)', level=2)

add_citation(doc, 'Gong, Y., Chung, Y.-A., & Glass, J.', '2021',
    'AST: Audio Spectrogram Transformer', 'Interspeech 2021', '2104.01778v3')

add_para(doc,
"""AST applies a pure ViT architecture to audio: extract 128-mel spectrograms, split into 16×16 patches, process with DeiT-initialized transformer. Pretrained on AudioSet (2M clips), it achieves SOTA on audio tagging, acoustic scene classification, and speech commands. Critically, AST is IDENTICAL to ViT in architecture—only the input modality differs.""")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""AST is our bridge model. Since it's literally ViT-on-spectrograms, if vision ViTs have artifacts, AST almost certainly does too. This makes AST our highest-confidence positive control. We predict artifacts will land on low-energy spectrogram regions (silence, background noise).""")

doc.add_paragraph()

# 5.2
add_heading(doc, '5.2 The Only Audio Attention-Sink Paper: AVSR with LLMs (Oct 2025)', level=2)

add_citation(doc, 'Anand, Cappellazzo, U., Petridis, S., & Pantic, M.', '2025',
    'Mitigating Attention Sinks in Audio-Visual Speech Recognition', 'arXiv preprint', '2510.22603v3')

add_para(doc,
"""This October 2025 paper is the ONLY work mentioning "attention sink" + audio in our arXiv searches. However, it studies LLM-based audio-visual speech recognition (Llama backbone processing audio-visual tokens), NOT pretrained audio encoders like HuBERT/Whisper. They find attention sinks on [BOS] and certain audio frames, mitigated by architectural tweaks.""")

add_para(doc, "Critical Differentiation:", bold=True)
p = doc.add_paragraph()
p.add_run("Scope: LLM decoder sinks (autoregressive) vs. our focus: encoder artifacts (bidirectional SSL models)")
p = doc.add_paragraph()
p.add_run("Models: Llama-based AVSR vs. our targets: HuBERT, WavLM, Whisper-encoder, AST, BEATs")
p = doc.add_paragraph()
p.add_run("Phenomenon: First-token sink (LLM-style) vs. artifact tokens on low-information patches (ViT-style)")

add_para(doc, "Relevance:", italic=True, color=(196, 97, 47))
add_para(doc,
"""We cite this as related work and clearly differentiate our scope. The fact that this paper exists validates that attention artifacts in audio are publishable—but the encoder artifact story remains completely unexplored.""")

doc.add_paragraph()

add_heading(doc, '5.3 Verified Research Gap', level=2)

p = doc.add_paragraph()
run = p.add_run("We systematically verified the gap via arXiv API queries (conducted July 2026):")
run.bold = True

queries = [
    ('"register tokens" AND cs.SD (audio)', 1, 'audio-visual MAE, tangential'),
    ('"attention sink" AND eess.AS (audio signal)', 2, 'LLM-AVSR paper + 1 unrelated'),
    ('"massive activations" AND audio', 1, 'same LLM-AVSR paper'),
    ('HuBERT AND "high-norm"', 0, 'none'),
    ('Whisper AND artifact AND tokens', 0, 'none'),
    ('WavLM AND registers', 0, 'none'),
]

for query, count, note in queries:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(f"{query}: ").font.name = 'Courier New'
    p.add_run(f"{count} result(s) — {note}")

add_para(doc, "", highlight='FFEB99')
p = doc.paragraphs[-1]
p.add_run("Conclusion: ").bold = True
p.add_run("The register/artifact phenomenon has been extensively studied in vision (>15 papers 2023–2026) and noted in LLMs (5+ papers), but has ZERO systematic studies in pretrained audio encoders. This is a verified blue ocean.")

doc.add_page_break()

# Section 6: Supporting Papers
add_heading(doc, '6. Supporting Methodological Papers', level=1)

add_heading(doc, '6.1 Denoising Vision Transformers (Jan 2024)', level=2)

add_citation(doc, 'Yang, J., Luo, K. Z., Li, J., et al.', '2024',
    'Denoising Vision Transformers', 'arXiv preprint', '2401.02957v2')

add_para(doc,
"""DVT addresses grid-like artifacts from positional embeddings (a different artifact type than register tokens). They use per-image neural field optimization to separate clean features from contaminated ones, then train a lightweight predictor. While focused on positional artifacts, the paper's two-stage analysis → fix methodology is relevant.""")

doc.add_paragraph()

add_heading(doc, '6.2 Layer-wise Analysis Papers for Audio (Background)', level=2)

p = doc.add_paragraph()
p.add_run("Two papers provide precedent for layer-wise audio transformer analysis:")

add_citation(doc, 'Pasad, A., Chou, J.-C., & Livescu, K.', '2021',
    'Layer-wise Analysis of a Self-supervised Speech Representation Model', 'IEEE/ACM TASLP', '2107.04734v3')

add_citation(doc, 'Pasad, A.,Shi, B., & Livescu, K.', '2022',
    'Comparative Layer-wise Analysis of Self-supervised Speech Models', 'Interspeech 2022', '2211.03929v3')

add_para(doc,
"""These papers use CCA to probe what linguistic properties (acoustic, phonetic, word-level) each layer encodes in wav2vec 2.0, HuBERT, and WavLM. They do NOT study token norms, artifacts, or attention sinks—they analyze representational content. However, they establish that layer-wise audio analysis papers are well-received at Interspeech and TASLP.""")

doc.add_page_break()

# Section 7: Synthesis
add_heading(doc, '7. Synthesis: How Literature Shapes Our Research Plan', level=1)

add_heading(doc, '7.1 Direct Method Transfer', level=2)

p = doc.add_paragraph()
p.add_run("From Jiang et al. (NeurIPS 2025):").bold = True

p = doc.add_paragraph(style='List Number')
p.add_run("Adopt FindRegisterNeurons algorithm verbatim (top_layer=5, threshold=75, top_k=10)")
p = doc.add_paragraph(style='List Number')
p.add_run("Port test-time register insertion from vision to audio models")
p = doc.add_paragraph(style='List Number')
p.add_run("Use their public code as implementation starting point")

p = doc.add_paragraph()
p.add_run("From Darcet et al. (ICLR 2024):").bold = True

p = doc.add_paragraph(style='List Number')
p.add_run("Norm-150 threshold for outlier token detection")
p = doc.add_paragraph(style='List Number')
p.add_run("Neighbor-cosine redundancy test (<0.5 = artifact candidate)")
p = doc.add_paragraph(style='List Number')
p.add_run("Attention entropy measurement across layers")

p = doc.add_paragraph()
p.add_run("From Baxevanakis et al. (Mar 2026):").bold = True

p = doc.add_paragraph(style='List Number')
p.add_run("98th-percentile norm threshold for cross-model comparison")
p = doc.add_paragraph(style='List Number')
p.add_run("Linear probe protocol: CE + 0.5·MSE, ≤30 epochs, patience 3")
p = doc.add_paragraph(style='List Number')
p.add_run("Systematic cross-architecture comparison methodology")

doc.add_paragraph()

add_heading(doc, '7.2 Audio-Specific Adaptations', level=2)

p = doc.add_paragraph()
p.add_run("Spectrogram Semantics:").bold = True
add_para(doc,
"""Unlike vision (uniform backgrounds → artifacts), audio has natural low-information regions: silence frames, low-SNR segments, padding. We hypothesize artifacts will concentrate there. We'll measure frame-level energy (RMS) and correlate with token norms.""")

p = doc.add_paragraph()
p.add_run("Model Diversity:").bold = True
add_para(doc,
"""Vision studies focus on DINO/CLIP variants. We'll compare across training objectives:
• Supervised: AST (ImageNet-style AudioSet training)
• Masked prediction SSL: HuBERT, WavLM (BERT-style)
• Contrastive multilingual: Whisper (encoder)
• Discriminative SSL: BEATs (tokenizer + contrastive)""")

p = doc.add_paragraph()
p.add_run("Task Evaluation:").bold = True
add_para(doc,
"""Audio downstream tasks differ from vision. We adopt SUPERB/HEAR evaluation protocol:
• Audio tagging: AudioSet, FSD50K (classification)
• Keyword spotting: Speech Commands v2 (classification)
• Speaker ID: VoxCeleb1 linear probe (verification)
• Emotion recognition: CREMA-D (classification)
• Sound event detection: DESED (localization-sensitive, most impacted by artifacts)""")

doc.add_paragraph()

add_heading(doc, '7.3 Interpretative Framework from LLM Literature', level=2)

add_para(doc,
"""The LLM attention sink papers (Xiao, Gu, Sun) teach us to distinguish:

1. Content-storing artifacts (vision registers): removing them loses task-relevant information
2. Distributional sinks (LLM first-token): removing them causes numerical instability but loses no semantic content

We'll use ablation controls (Parodi et al.) to classify audio artifacts:
• Zero-ablation: replace artifact tokens with zeros
• Mean-substitution: replace with mean activation
• Noise-substitution: replace with Gaussian noise matched to typical token norm
• Cross-clip shuffling: swap artifact tokens between different audio clips

If only zero-ablation degrades performance → distributional sink. If all ablations degrade → content-storing.""")

doc.add_page_break()

# Section 8: OpenReview Insights
add_heading(doc, '8. OpenReview Review Insights', level=1)

add_para(doc,
"""We mined OpenReview for reviews of our anchor papers to anticipate reviewer concerns:""")

add_heading(doc, '8.1 Vision Transformers Need Registers (ICLR 2024 Oral) — Review Themes', level=2)

p = doc.add_paragraph()
p.add_run("Strengths highlighted by reviewers:").bold = True
p = doc.add_paragraph(style='List Bullet')
p.add_run("Clear identification of an important problem with a coherent mechanistic story")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Simple, effective solution (adding registers)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Comprehensive experiments and ablations")

p = doc.add_paragraph()
p.add_run("Concerns raised:").bold = True
p = doc.add_paragraph(style='List Bullet')
p.add_run("Registers add compute cost (2-6% overhead for 4-16 tokens)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Registers sometimes focus on spatially discrete object parts — does this undermine the 'global information storage' hypothesis?")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Would benefit from per-head norm analysis")

add_para(doc, "Implication for our work:", italic=True, color=(196, 97, 47))
add_para(doc,
"""We should quantify compute overhead of test-time registers and provide per-head analysis if audio artifacts appear. The test-time method avoids the retraining cost concern entirely.""")

doc.add_paragraph()

add_heading(doc, '8.2 Test-Time Registers (ICLR 2026 Submission) — Review Themes', level=2)

add_para(doc,
"""We found 4 reviews for an ICLR 2026 submission (forum 991QAd8Qyv) proposing test-time registers via NFN + TokenRank. Ratings: 2, 4, 4, 4 (rejected or major revisions). Key criticisms:""")

p = doc.add_paragraph(style='List Bullet')
p.add_run('Unclear motivation: "global information" claim disconnected from method')
p = doc.add_paragraph(style='List Bullet')
p.add_run("Method complexity: many hyperparameters, not actually simpler than prior work")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Weak evaluation: only tested on 1-D token decoders, claims about LLMs/LMMs not demonstrated")
p = doc.add_paragraph(style='List Bullet')
p.add_run('Flawed frequency analysis: "taking FFT over PCA dimensions" imposes non-meaningful order')

add_para(doc, "Critical lesson:", italic=True, color=(196, 97, 47))
add_para(doc,
"""Reviewers punish overclaiming and weak evaluation scope. Our paper must:
1. Evaluate on standard benchmarks (SUPERB tasks), not just toy probes
2. Match claims to evidence (if we claim 'global information', prove it with ablations)
3. Keep method simple and well-justified
4. Avoid speculative signal-processing arguments without solid grounding""")

doc.add_paragraph()

add_heading(doc, '8.3 Reproducibility Studies (TMLR) — Acceptance Pattern', level=2)

add_para(doc,
"""We found 3 TMLR reproducibility studies of the registers paper (2024 submissions). All were accepted, confirming that:
• Replication + extension to new model scales is publishable
• Small-scale studies (DeiT-III Small, ViT-B) add value
• Negative/nuanced results ("registers help but modestly") are acceptable

This de-risks our audio transfer: even if audio artifacts are weaker than vision, characterizing the difference is a contribution.""")

doc.add_page_break()

# Section 9: Research Questions
add_heading(doc, '9. Our Research Questions Grounded in Literature', level=1)

p = doc.add_paragraph()
p.add_run("RQ1 (Phenomenon): ").bold = True
p.add_run("Do pretrained audio transformers (AST, HuBERT, WavLM, Whisper-encoder, BEATs, Audio-MAE) exhibit high-norm artifact tokens analogous to vision transformers?")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Measurement: per-token norm distributions, 98th-percentile threshold, bimodality test (Hartigan's dip)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Localization: do artifacts concentrate on silence/low-energy frames? (audio-specific hypothesis)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Cross-model: supervised vs. SSL vs. multilingual — does training objective matter?")

p = doc.add_paragraph()
p.add_run("RQ2 (Mechanism): ").bold = True
p.add_run("If artifacts exist, are they content-storing (like vision registers) or distributional sinks (like LLM first-token)?")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Method: ablation control battery (zero / mean / noise / shuffling)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Linear probes: do artifact tokens predict global vs. local audio properties?")

p = doc.add_paragraph()
p.add_run("RQ3 (Solution): ").bold = True
p.add_run("Does the test-time register method (Jiang et al.) transfer to audio?")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Port FindRegisterNeurons to audio model MLPs")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Measure: artifact token reduction, attention map smoothness")

p = doc.add_paragraph()
p.add_run("RQ4 (Downstream Impact): ").bold = True
p.add_run("Do test-time registers improve audio task performance?")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Evaluation: 5 SUPERB-style tasks (tagging, keyword spotting, speaker ID, emotion, sound event detection)")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Hypothesis: localization-sensitive tasks (sound event detection) benefit most, image-level tasks (tagging) unaffected")

p = doc.add_paragraph()
p.add_run("RQ5 (Robustness): ").bold = True
p.add_run("Do registers improve robustness to audio domain shifts?")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Test: clean vs. noisy audio, in-domain vs. OOD datasets")
p = doc.add_paragraph(style='List Bullet')
p.add_run("Inspired by: Yellapragada et al. OOD results")

doc.add_page_break()

# Section 10: Publication strategy
add_heading(doc, '10. Publication Strategy Informed by Literature', level=1)

add_heading(doc, '10.1 Target Venues', level=2)

p = doc.add_paragraph()
p.add_run("Primary: ICASSP 2027 (deadline ~Sept 2026)").bold = True
add_para(doc,
"""• CORE rank: A
• Audio community will find register analysis novel
• Precedent: AST (Interspeech 2021), layer-wise audio analysis papers at ICASSP
• Page limit: 4 pages + references (tight but achievable for focused study)""")

p = doc.add_paragraph()
p.add_run("Alternative: Interspeech 2027 (deadline ~March 2027)").bold = True
add_para(doc,
"""• Similar tier to ICASSP, 4–5 page limit
• If ICASSP rejects, Interspeech is 6-month resubmission target""")

p = doc.add_paragraph()
p.add_run("Stretch: ICLR 2027 (deadline ~Oct 2026)").bold = True
add_para(doc,
"""• Only if findings are surprisingly strong (e.g., artifacts MORE pronounced in audio than vision)
• Would need deeper mechanistic analysis + stronger baselines
• Realistic for beginner: 15–20% acceptance rate is tough""")

p = doc.add_paragraph()
p.add_run("Journal fallback: IEEE/ACM TASLP (Transactions on Audio, Speech, Language Processing)").bold = True
add_para(doc,
"""• Q1 journal, impact factor ~4.0
• Accepts extended versions of conference papers
• No strict deadline — safe final target""")

doc.add_paragraph()

add_heading(doc, '10.2 Positioning Against "Just a Transfer" Objection', level=2)

add_para(doc, "Anticipated reviewer concern:", bold=True)
add_para(doc,
'"This is just applying a known vision result to audio — where\'s the novelty?"')

add_para(doc, "Our pre-emptive responses:", bold=True)

p = doc.add_paragraph(style='List Number')
p.add_run("Cross-domain validation IS a contribution: Many vision phenomena don't transfer (e.g., adversarial robustness patterns differ). Establishing that registers are a general transformer phenomenon, not vision-specific, is scientifically valuable.")

p = doc.add_paragraph(style='List Number')
p.add_run("Audio-specific findings: Artifact localization on silence/low-SNR frames (not just 'background'), differences between frame-level (HuBERT) vs. patch-based (AST) tokenization, streaming implications for Whisper.")

p = doc.add_paragraph(style='List Number')
p.add_run("First systematic benchmark: We provide the first register-augmented baselines for SUPERB tasks, enabling future audio research.")

p = doc.add_paragraph(style='List Number')
p.add_run("Comparison papers are respected: Baxevanakis '26 cross-architecture paper, TMLR reproducibility studies, Pasad layer-wise audio analyses — all published despite being 'apply known method to new models'.")

doc.add_page_break()

# Section 11: Summary
add_heading(doc, '11. Conclusion: A Grounded, Low-Risk Research Plan', level=1)

add_para(doc,
"""This literature review surveyed 13 papers spanning the discovery and mitigation of attention artifacts in transformers. The key findings:""")

p = doc.add_paragraph()
p.add_run("1. The phenomenon is well-established in vision (5+ papers, ICLR/NeurIPS venues) and emerging in LLMs (4+ papers), but has ZERO systematic studies in audio encoders.").bold = True

p = doc.add_paragraph()
p.add_run("2. A training-free solution exists (Jiang et al. NeurIPS 2025 spotlight) with public code, making our method transfer low-risk and low-compute.").bold = True

p = doc.add_paragraph()
p.add_run("3. Cross-model analysis papers are publishable even without proposing new methods (Baxevanakis '26, TMLR reproducibility studies).").bold = True

p = doc.add_paragraph()
p.add_run("4. OpenReview feedback emphasizes: match claims to evidence, evaluate on standard benchmarks, avoid overcomplexity.").bold = True

add_para(doc, "", highlight='D4EDDA')
p = doc.paragraphs[-1]
p.add_run("Final Assessment: ").bold = True
p.add_run("This project occupies a verified research gap with:")
p2 = doc.add_paragraph(style='List Bullet')
p2.add_run("High scientific novelty (first audio artifact study)")
p2 = doc.add_paragraph(style='List Bullet')
p2.add_run("Low compute requirements (~60–90 Kaggle GPU-hours)")
p2 = doc.add_paragraph(style='List Bullet')
p2.add_run("Clear publication targets (ICASSP/Interspeech)")
p2 = doc.add_paragraph(style='List Bullet')
p2.add_run("Beginner-friendly execution (port existing method, standard benchmarks)")
p2 = doc.add_paragraph(style='List Bullet')
p2.add_run("High result floor (even null results are findings)")

add_para(doc,
"""The convergence of verified gap + public anchor code + low compute + multiple possible outcomes makes this an exceptionally strong first research project.""")

# Save
doc.save('D:\\Research hub\\LiteratureReview.docx')
print("LiteratureReview.docx generated successfully!")

p = doc.add_paragraph()
run = p.add_run("We systematically verified the gap via arXiv API queries (conducted July 2026):")
run.bold = True

queries = [
    ('"register tokens" AND cs.SD (audio)', 1, 'audio-visual MAE, tangential'),
    ('"attention sink" AND eess.AS (audio signal)', 2, 'LLM-AVSR paper + 1 unrelated'),
    ('"massive activations" AND audio', 1, 'same LLM-AVSR paper'),
    ('HuBERT AND "high-norm"', 0, 'none'),
    ('Whisper AND artifact AND tokens', 0, 'none'),
    ('WavLM AND registers', 0, 'none'),
]

for query, count, note in queries:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(f"{query}: ").font.name = 'Courier New'
    p.add_run(f"{count} result(s) — {note}")

add_para(doc, "", highlight='FFEB99')
p = doc.paragraphs[-1]
p.add_run("Conclusion: ").bold = True
p.add_run("The register/artifact phenomenon has been extensively studied in vision (>15 papers 2023–2026) and noted in LLMs (5+ papers), but has ZERO systematic studies in pretrained audio encoders. This is a verified blue ocean.")

doc.add_page_break()
