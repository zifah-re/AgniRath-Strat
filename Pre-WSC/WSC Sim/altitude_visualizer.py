"""Visualize the WSC route's altitude profile from a .kml.save route file.

Usage:
    python visualize_altitude.py                      # auto-detects the route file
    python visualize_altitude.py "path/to/route.kml.save"
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Same control-stop markers as optimizer.py, just for visual reference on the plot.
CONTROL_STOPS_KM = {
    "Katherine": 318.0, "Dunmarra": 635.0, "Tennant Creek": 988.0,
    "Barrow Creek": 1210.0, "Alice Springs": 1493.0, "Erldunda": 1693.0,
    "Coober Pedy": 2179.0, "Glendambo": 2432.0, "Port Augusta": 2719.0,
}


def find_route_file() -> Path:
    """An explicit path on the command line, the exact name optimizer.py expects,
    or (as a fallback) the first *.kml.save file sitting next to this script."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            return path
        raise FileNotFoundError(f"No such file: {path}")

    default = HERE / "WSC_WSC Route_WSC Route.kml.save"
    if default.exists():
        return default

    candidates = sorted(HERE.glob("*.kml.save"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Couldn't find a .kml.save route file next to this script. Pass a path "
        'explicitly:\n    python visualize_altitude.py "path/to/route.kml.save"'
    )


def load_altitude_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    profile = json.loads(path.read_text(encoding="utf-8"))["profile"]
    distance_km = np.asarray(profile["Distance"], dtype=float)
    altitude_m = np.asarray(profile["Altitude"], dtype=float)

    # Defensive, same spirit as optimizer.py's own `aligned()` helper --
    # these two arrays should already be the same length, but just in case.
    n = min(len(distance_km), len(altitude_m))
    return distance_km[:n], altitude_m[:n]


def plot_altitude(distance_km: np.ndarray, altitude_m: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(distance_km, altitude_m, color="saddlebrown", linewidth=1.2)
    ax.fill_between(distance_km, altitude_m, altitude_m.min(), color="saddlebrown", alpha=0.15)

    for name, stop_km in CONTROL_STOPS_KM.items():
        if distance_km[0] <= stop_km <= distance_km[-1]:
            ax.axvline(stop_km, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.text(stop_km, altitude_m.max(), name, rotation=90, va="top", ha="right", fontsize=7)

    ax.set(
        xlim=(distance_km[0], distance_km[-1]),
        xlabel="Cumulative distance (km)",
        ylabel="Altitude (m)",
        title="WSC Route Altitude Profile",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[*] Saved plot to: {out_path}")
    plt.show()


def main() -> None:
    route_path = find_route_file()
    print(f"[*] Reading route file: {route_path.name}")
    distance_km, altitude_m = load_altitude_profile(route_path)
    print(
        f"[*] {len(distance_km)} points | {distance_km[-1]:.1f} km total | "
        f"altitude range {altitude_m.min():.0f}-{altitude_m.max():.0f} m"
    )

    out_path = route_path.with_name("wsc_altitude_profile.png")
    plot_altitude(distance_km, altitude_m, out_path)


if __name__ == "__main__":
    main()