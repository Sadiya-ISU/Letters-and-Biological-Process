"""Fail if any {{token}} remains unresolved in the manuscript deliverables."""
import re, sys
from pathlib import Path

MR = Path(__file__).resolve().parents[2] / "manuscript_revision"
ALLOWED_TODO = {"{{ZENODO_DOI}}"}  # filled only after the Zenodo deposit
SKIP_FILES = {"numbers_crosscheck.md", "NUMBERS.md"}
bad = []
for f in MR.glob("*.md"):
    if f.name in SKIP_FILES:
        continue
    for tok in re.findall(r"\{\{[^}]+\}\}", f.read_text()):
        if tok not in ALLOWED_TODO:
            bad.append((f.name, tok))
if bad:
    print("UNRESOLVED TOKENS:")
    for fn, t in bad:
        print(f"  {fn}: {t}")
    sys.exit(1)
print("All tokens resolved (except the post-deposit Zenodo DOI).")
