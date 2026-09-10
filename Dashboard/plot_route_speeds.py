"""Plot the speed limit and traffic-speed profile from a dashboard .kml.save.

Example:
    python plot_route_speeds.py "Saves/2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop.kml.save"

Optionally save an image instead of (or as well as) opening the plot:
    python plot_route_speeds.py "Saves/example.kml.save" --output route_speeds.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_speed_profiles(save_path: Path) -> tuple[list[float], list[float], list[float]]:
    """Read the dashboard's JSON-based .kml.save file and validate its arrays."""
    try:
        saved_data = json.loads(save_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Save file not found: {save_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{save_path} is not valid dashboard save JSON: {error}") from error

    profile = saved_data.get("profile") or {}
    distance = profile.get("Distance") or []
    speed_limit = profile.get("SpeedLimit") or []
    speed_profile = profile.get("SpeedProfile") or []

    if not distance:
        raise SystemExit("The save has no profile.Distance values.")

    expected = len(distance)
    mismatches = {
        "SpeedLimit": len(speed_limit),
        "SpeedProfile": len(speed_profile),
    }
    wrong_lengths = [f"{name}={length}" for name, length in mismatches.items() if length != expected]
    if wrong_lengths:
        raise SystemExit(
            f"Profile arrays must match Distance ({expected} points); found "
            + ", ".join(wrong_lengths)
            + "."
        )

    return list(map(float, distance)), list(map(float, speed_limit)), list(map(float, speed_profile))


def plot_speed_profiles(save_path: Path, output_path: Path | None = None) -> None:
    distance, speed_limit, speed_profile = load_speed_profiles(save_path)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(distance, speed_limit, label="Speed limit", color="#ef4444", linewidth=1.8)
    axis.plot(distance, speed_profile, label="Traffic speed profile", color="#3b82f6", linewidth=1.3)
    axis.set_title(f"Speed profiles — {save_path.stem}")
    axis.set_xlabel("Distance (km)")
    axis.set_ylabel("Speed (km/h)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=180)
        print(f"Saved plot to {output_path.resolve()}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot speed profiles from a dashboard .kml.save file.")
    parser.add_argument("save_file", type=Path, help="Path to the dashboard .kml.save file")
    parser.add_argument("--output", "-o", type=Path, help="Optional output PNG/PDF/SVG path")
    arguments = parser.parse_args()
    plot_speed_profiles(arguments.save_file, arguments.output)


if __name__ == "__main__":
    main()
