"""Power-balance equation with hand-written labels, transparent PNG for the dark slides."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent / "deck_fig"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "cm"})
WHITE, ORANGE, HAND = "#F4F2EE", "#FF7A1A", "#F1C9A8"


def equation(terms, name, size=(16, 3.4), fs=34, y=0.68, x0=0.01):
    fig = plt.figure(figsize=size, dpi=150)
    fig.patch.set_alpha(0)
    r = fig.canvas.get_renderer()
    x = x0
    W = fig.get_figwidth() * fig.dpi
    for i, (tex, lab, col) in enumerate(terms):
        t = fig.text(x, y, tex, fontsize=fs, color=col, va="center", ha="left")
        bb = t.get_window_extent(r)
        cx = (bb.x0 + bb.x1) / 2 / W
        if lab:
            ly = 0.16 if i % 2 == 0 else 0.30
            fig.text(cx, ly, lab, fontsize=30, color=HAND, family="Caveat", ha="center", va="center")
            a = FancyArrowPatch((cx, ly + 0.07), (cx, y - 0.16), transform=fig.transFigure,
                                arrowstyle="-|>", mutation_scale=18, color=HAND, lw=1.6,
                                connectionstyle="arc3,rad=-0.25")
            fig.patches.append(a)
        x = bb.x1 / W + 0.006
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)
    from PIL import Image
    im = Image.open(OUT / name)
    x0, y0, x1, y1 = im.getbbox()
    im.crop((max(x0 - 10, 0), max(y0 - 10, 0), min(x1 + 10, im.width), min(y1 + 10, im.height))).save(OUT / name)
    print(name, round(x, 3))


if __name__ == "__main__":
    equation([
        (r"$P_{\mathrm{bat}}$", "", ORANGE),
        (r"$=$", "", WHITE),
        (r"$\eta_{\mathrm{array}}\,A\,I$", "sun in", WHITE),
        (r"$-\;(\frac{1}{2}\rho\, C_dA\, v_{\mathrm{air}}^2$", "drag", WHITE),
        (r"$+\;C_{rr}\, m g$", "tyres", WHITE),
        (r"$+\;m g \sin\theta$", "hills", WHITE),
        (r"$+\;m a)\, v$", "speeding up", WHITE),
        (r"$/\;\eta_{\mathrm{drive}}$", "drivetrain loss", ORANGE),
        (r"$-\;P_{\mathrm{idle}}$", "electronics", WHITE),
    ], "eq_power.png")
    equation([
        (r"$E_{\mathrm{bus}}\,\eta$", "energy at the wheel", ORANGE),
        (r"$=$", "", WHITE),
        (r"$C_{rr}\, m g D$", "tyres", WHITE),
        (r"$+\;C_dA\sum \frac{1}{2}\rho\, v_{\mathrm{air}}^2\, v\,\Delta t$", "drag, real air + wind", WHITE),
        (r"$+\;m g\,\Delta h$", "climb", WHITE),
        (r"$+\;\Delta KE$", "speed change", WHITE),
    ], "eq_energy.png", size=(12.5, 3.4), fs=31, x0=0.07)
