"""Crop legend and panel regions of the three replotted PNGs and list their most common colours."""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

src = Path(sys.argv[1])
out = Path(sys.argv[2])
jobs = {
    "match_vs_resonator_count.png": [("f1_legend", (60, 60, 360, 215)), ("f1_jump", (300, 50, 520, 520))],
    "match_vs_resonators_all_fits.png": [("f2_legend", (40, 835, 1400, 960)), ("f2_panel12", (360, 30, 720, 420))],
    "ring_time_vs_hold_time.png": [("f3_legend", (280, 1090, 1120, 1160)), ("f3_panel_clean", (690, 570, 1250, 1090))],
}
for name, boxes in jobs.items():
    im = Image.open(src / name).convert("RGB")
    print(name, im.size)
    cnt = Counter(im.getdata())
    print("  top colours:", ", ".join(f"#{r:02x}{g:02x}{b:02x}:{n}" for (r, g, b), n in cnt.most_common(14)))
    for tag, box in boxes:
        c = im.crop(box)
        c = c.resize((c.width * 2, c.height * 2), Image.LANCZOS)
        c.save(out / f"{tag}.png")
