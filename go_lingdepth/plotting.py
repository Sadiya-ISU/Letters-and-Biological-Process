"""Shared style + alt-text registry so every figure ships accessibility text."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALT_TEXT: dict[str, str] = {}


def register_alt(stem: str, text: str):
    ALT_TEXT[stem] = text


def save(fig, figures_dir, stem: str, alt: str):
    register_alt(stem, alt)
    fig.savefig(f"{figures_dir}/{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{figures_dir}/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def dump_alt_text(figures_dir):
    lines = ["# Figure alt-text\n"]
    for stem, txt in ALT_TEXT.items():
        lines.append(f"## {stem}\n\n{txt}\n")
    open(f"{figures_dir}/alt_text.md", "w").write("\n".join(lines))
