"""LaTeX-style (mathtext) equations for the report, black on white."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parent / "fig"
plt.rcParams.update({"mathtext.fontset": "cm"})


def eq(tex, name, fs=26):
    fig = plt.figure(figsize=(12, 1.2), dpi=200)
    fig.text(0.5, 0.5, tex, fontsize=fs, ha="center", va="center", color="#111111")
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    im = Image.open(OUT / name).convert("RGB")
    bg = Image.new("RGB", im.size, "white")
    from PIL import ImageChops
    x0, y0, x1, y1 = ImageChops.difference(im, bg).getbbox()
    im.crop((x0 - 20, y0 - 20, x1 + 20, y1 + 20)).save(OUT / name)


eq(r"$P_{\mathrm{bat}} = \eta_{\mathrm{array}}\,A\,I \;-\; \dfrac{\left(\frac{1}{2}\rho\,C_dA\,v_{\mathrm{air}}^2 + C_{rr}\,m g + m g\sin\theta + m a\right) v}{\eta_{\mathrm{drive}}} \;-\; P_{\mathrm{idle}}$", "eq_power.png")
eq(r"$E_{\mathrm{bus}}\,\eta_{\mathrm{drive}} = C_{rr}\,m g D \;+\; C_dA \sum \frac{1}{2}\rho\, v_{\mathrm{air}}^2\, v\,\Delta t \;+\; m g\,\Delta h \;+\; \Delta KE$", "eq_energy.png")
print("ok")
