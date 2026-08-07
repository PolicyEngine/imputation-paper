"""Generate the Populace pipeline overview figure.

A "snapshot of what Populace can do": the stages a microdata population goes
through prior to and including calibration, the package that owns each stage,
and the single ``populace.frame.Frame`` carried through all of them.

Outputs (written next to this file):
- ``populace_pipeline.png`` (slide-ready, 200 dpi)

Run:
    uv run --with matplotlib python paper/figures/populace_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- PolicyEngine palette ------------------------------------------------
BLUE = "#2C6496"      # primary
DARK = "#17354F"      # darkest blue (text / spine)
TEAL = "#39C6C0"      # accent
GREY = "#808080"
LIGHT = "#F2F6FA"     # stage fill
LIGHTER = "#E8F0F7"
GOLD = "#FFC72C"      # highlight the calibration stage (paper focus)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # DejaVu Sans first: it has full glyph coverage (the arrows render).
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "svg.fonttype": "none",
    }
)

# Stages, prior to and INCLUDING calibration. Each: (title, package, sublines).
STAGES = [
    (
        "Sources",
        "populace-data",
        ["CPS ASEC (spine)", "IRS PUF, ACS,", "other surveys"],
    ),
    (
        "Combine",
        "populace-frame",
        ["entity tables", "+ typed weights", "+ strata"],
    ),
    (
        "Impute",
        "populace-fit",
        ["weighted QRF", "chained draws", "fills tax / assets"],
    ),
    (
        "Geography",
        "populace-build",
        ["assign sub-national", "areas; records may", "appear in many"],
    ),
    (
        "Build targets",
        "populace-calibrate",
        ["admin totals →", "sparse matrix M", "(PE simulation)"],
    ),
    (
        "Calibrate",
        "populace-calibrate",
        ["L0 / Hard-Concrete", "gates + log-weights", "generate-big-then-prune"],
    ),
]

# Geometry --------------------------------------------------------------
N = len(STAGES)
BOX_W, BOX_H = 1.9, 1.58
GAP = 0.68
X0 = 0.5
Y_BOX = 2.7
fig_w = X0 * 2 + N * BOX_W + (N - 1) * GAP
fig_h = 6.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")


def box_x(i: int) -> float:
    return X0 + i * (BOX_W + GAP)


# Title -----------------------------------------------------------------
ax.text(
    fig_w / 2,
    fig_h - 0.42,
    "The Populace pipeline",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=DARK,
)
ax.text(
    fig_w / 2,
    fig_h - 0.92,
    "from source surveys to a calibrated microdata population — one Frame, packages as operators",
    ha="center",
    va="center",
    fontsize=12.5,
    color=GREY,
)

# Stage boxes -----------------------------------------------------------
for i, (title, pkg, subs) in enumerate(STAGES):
    x = box_x(i)
    is_focus = title == "Calibrate"
    edge = GOLD if is_focus else BLUE
    fill = "#FFF8E6" if is_focus else LIGHT
    lw = 2.6 if is_focus else 1.4

    box = FancyBboxPatch(
        (x, Y_BOX),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(box)

    # step number chip
    ax.add_patch(
        mpatches.Circle(
            (x + 0.26, Y_BOX + BOX_H - 0.21),
            0.13,
            facecolor=edge if is_focus else BLUE,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        x + 0.26,
        Y_BOX + BOX_H - 0.21,
        str(i + 1),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=DARK if is_focus else "white",
        zorder=5,
    )

    ax.text(
        x + BOX_W / 2,
        Y_BOX + BOX_H - 0.43,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        zorder=5,
    )
    for k, line in enumerate(subs):
        ax.text(
            x + BOX_W / 2,
            Y_BOX + BOX_H - 0.78 - k * 0.30,
            line,
            ha="center",
            va="center",
            fontsize=9.8,
            color="#37474F",
            zorder=5,
        )

    # package label under the box
    ax.text(
        x + BOX_W / 2,
        Y_BOX - 0.27,
        pkg,
        ha="center",
        va="center",
        fontsize=9.3,
        style="italic",
        color=BLUE if not is_focus else "#B8860B",
        fontweight="bold",
        zorder=5,
    )

    # arrow to next stage
    if i < N - 1:
        ax.add_patch(
            FancyArrowPatch(
                (x + BOX_W, Y_BOX + BOX_H / 2),
                (x + BOX_W + GAP, Y_BOX + BOX_H / 2),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color=GREY,
                zorder=2,
            )
        )

# The Frame spine -------------------------------------------------------
spine_y = 1.35
spine_x0 = box_x(0)
spine_x1 = box_x(N - 1) + BOX_W
spine = FancyBboxPatch(
    (spine_x0, spine_y - 0.42),
    spine_x1 - spine_x0,
    0.84,
    boxstyle="round,pad=0.01,rounding_size=0.1",
    linewidth=1.4,
    edgecolor=DARK,
    facecolor=LIGHTER,
    zorder=1,
)
ax.add_patch(spine)
ax.text(
    (spine_x0 + spine_x1) / 2,
    spine_y + 0.13,
    "populace.frame.Frame  —  one weighted sampling frame carried through every stage",
    ha="center",
    va="center",
    fontsize=12.2,
    fontweight="bold",
    color=DARK,
    zorder=2,
)
ax.text(
    (spine_x0 + spine_x1) / 2,
    spine_y - 0.18,
    "typed weights (design → importance → calibrated)  ·  strata / provenance  ·  links  ·  variable metadata  ·  weighted accounting",
    ha="center",
    va="center",
    fontsize=9.4,
    color="#37474F",
    zorder=2,
)
# thin connectors from each box down to the spine
for i in range(N):
    x = box_x(i) + BOX_W / 2
    ax.plot([x, x], [Y_BOX, spine_y + 0.42], color=GREY, lw=0.7, ls=":", zorder=0)

# Candidate-universe size annotation (generate-big-then-prune) ----------
band_y = Y_BOX + BOX_H + 0.55
growth_start = box_x(0)
growth_end = box_x(4) + BOX_W
prune_start = box_x(5)
prune_end = box_x(5) + BOX_W
prune_arrow_start = growth_end
ax.annotate(
    "",
    xy=(growth_end, band_y),
    xytext=(growth_start, band_y),
    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2),
)
ax.text(
    (growth_start + growth_end) / 2,
    band_y + 0.22,
    "candidate universe grows   ~300k → 3M → 30M records",
    ha="center",
    va="bottom",
    fontsize=10.2,
    color="#1A8F8A",
    fontweight="bold",
)
ax.annotate(
    "",
    xy=(prune_end, band_y),
    xytext=(prune_arrow_start, band_y),
    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.2),
)
ax.text(
    (prune_start + prune_end) / 2,
    band_y + 0.22,
    "pruned to budget",
    ha="center",
    va="bottom",
    fontsize=10.2,
    color="#B8860B",
    fontweight="bold",
)

# Output: calibrated dataset -------------------------------------------
ax.add_patch(
    FancyArrowPatch(
        (spine_x1, spine_y),
        (spine_x1 + 0.001, spine_y),
        arrowstyle="-|>",
        mutation_scale=1,
        color="none",
    )
)
ax.text(
    fig_w / 2,
    0.34,
    "Output:  retained records + calibrated weights  →  PolicyEngine-ready dataset "
    "(gates evaluated deterministically; carries an environment certificate)",
    ha="center",
    va="center",
    fontsize=10,
    color=GREY,
)

out_dir = Path(__file__).resolve().parent
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(out_dir / "populace_pipeline.png", dpi=200, bbox_inches="tight")
print(f"wrote {out_dir / 'populace_pipeline.png'}")
