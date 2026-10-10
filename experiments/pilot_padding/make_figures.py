"""Figures for the paper draft from pilot JSON results (experiments/pilot_padding/results)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "experiments/pilot_padding/results"; OUT = "paper/figures"
models = [("whisper-base", "Whisper-base"), ("whisper-small", "Whisper-small"), ("whisper-large-v3-turbo", "Whisper-large-v3-turbo")]
res = {m: json.load(open(os.path.join(R, f"{m}_padkv_n100.json"))) for m, _ in models}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# Fig: TOP-k vs RND-k cached silence frames
ks = [4, 16, 64, 256]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
for ax, (m, name) in zip(axes, models):
    r = res[m]["results"]
    ax.plot(ks, [r[f"kvTOP{k}+tail"]["wer"] for k in ks], "o-", color="#1f5fa8", label="top-attended silence frames")
    ax.plot(ks, [r[f"kvRND{k}+tail"]["wer"] for k in ks], "s--", color="#c0504d", label="random silence frames")
    ax.axhline(r["pad30"]["wer"], color="k", lw=0.8, label="30-s padding")
    ax.axhline(r["tail"]["wer"], color="gray", lw=0.8, ls=":", label="silence tail only")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_title(name); ax.set_xlabel("cached silence frames $k$")
axes[0].set_ylabel("WER (%)")
axes[2].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "topk_vs_random.pdf")); plt.close(fig)

# Fig: per-layer max/median norm ratio, silence-only input
fig, ax = plt.subplots(figsize=(3.3, 2.1))
for (m, name), c in zip(models, ["#1f5fa8", "#4a9a4a", "#c0504d"]):
    v = res[m]["silence_norm_ratio_by_layer"]
    x = [i / (len(v) - 1) for i in range(len(v))]
    ax.plot(x, v, "o-", ms=2.5, color=c, label=name)
ax.set_yscale("log"); ax.set_xlabel("relative depth (block output)"); ax.set_ylabel("max / median token norm")
ax.legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(os.path.join(OUT, "norm_ratio_silence.pdf")); plt.close(fig)

# Fig: SilenceCache schematic
fig, ax = plt.subplots(figsize=(7.0, 2.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec="k", lw=0.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.6)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=8, lw=0.7))
NL = chr(10)
box(1, 22, 15, 8, "30 s digital silence" + NL + "(offline, once)", "#eeeeee")
box(21, 22, 19, 8, "Whisper encoder" + NL + "(all 1500 positions)", "#dde8f5")
box(45, 22, 27, 8, "cache per layer: K/V of positions [T, 1500)" + NL + "+ encoder outputs (silence tail)", "#fdeccf")
box(1, 3, 15, 8, "utterance" + NL + "(T positions, no padding)", "#eeeeee")
box(21, 3, 19, 8, "Whisper encoder on T positions;" + NL + "queries attend to own + cached K/V", "#dde8f5")
box(45, 3, 27, 8, "[speech outputs ; cached silence tail]", "#e3f1df")
box(78, 3, 20, 8, "Whisper decoder" + NL + "(unchanged)", "#dde8f5")
arr(16.6, 26, 20.4, 26); arr(40.6, 26, 44.4, 26); arr(16.6, 7, 20.4, 7); arr(40.6, 7, 44.4, 7); arr(72.6, 7, 77.4, 7)
arr(52, 21.4, 34, 11.6); arr(64, 21.4, 62, 11.6)
ax.text(40.5, 16.5, "cached K/V", fontsize=6.5); ax.text(63.5, 16.5, "cached tail", fontsize=5.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "silencecache_overview.pdf")); plt.close(fig)
print("figures written")
