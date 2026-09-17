"""Create a mirrored out-and-back .kml.save from a dashboard .kml.save route.

The dashboard save's profile.Coordinates are (latitude, longitude).

The generated file remains a dashboard-compatible .kml.save JSON file.
The route is cut at half of the requested total loop distance, then mirrored
back to the original start point.

Example:

    python make_mirrored_loop_kml.py ^
        "Saves/2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop.kml.save" ^
        34 ^
        --output "Saves/mirrored_34km.kml.save"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_save(save_path: Path) -> dict:
    """Load the complete dashboard .kml.save JSON."""
    try:
        return json.loads(save_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Save file not found: {save_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Not valid dashboard save JSON: {save_path} ({exc})"
        ) from exc


def load_profile(
    saved: dict,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Load Distance and Coordinates from the dashboard profile."""
    profile = saved.get("profile") or {}

    distances = profile.get("Distance") or []
    coordinates = profile.get("Coordinates") or []

    if len(distances) != len(coordinates) or len(distances) < 2:
        raise SystemExit(
            "profile.Distance and profile.Coordinates must both exist, "
            "have equal lengths, and contain at least two points."
        )

    try:
        distances_km = [float(distance) for distance in distances]

        coords = [
            (float(lat), float(lon))
            for lat, lon in coordinates
        ]

    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Distances or coordinates contain non-numeric values."
        ) from exc

    return distances_km, coords


def route_half_at_distance(
    distances_km: list[float],
    coordinates: list[tuple[float, float]],
    total_loop_distance_km: float,
) -> tuple[list[tuple[float, float]], int, float]:
    """Cut the original route at half of the requested total loop distance."""

    if total_loop_distance_km <= 0:
        raise ValueError("Total loop distance must be greater than zero.")

    half_distance_km = total_loop_distance_km / 2.0

    closest_index = min(
        range(len(distances_km)),
        key=lambda index: abs(
            distances_km[index] - half_distance_km
        ),
    )

    return (
        coordinates[: closest_index + 1],
        closest_index,
        distances_km[closest_index],
    )


def mirror_route(
    outbound_coordinates: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Create the return leg by reversing the outbound route."""

    if len(outbound_coordinates) < 2:
        raise ValueError(
            "At least two outbound coordinates are required to mirror a route."
        )

    # Keep the turn-around point only once.
    return outbound_coordinates + outbound_coordinates[-2::-1]


def build_mirrored_distances(
    mirrored_coordinates: list[tuple[float, float]],
    outbound_distances: list[float],
) -> list[float]:
    """Build cumulative distance values for the mirrored route.

    Distances are recalculated from the coordinate geometry so that the
    generated .kml.save has a consistent Distance array.
    """

    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        lat1, lon1 = a
        lat2, lon2 = b

        r = 6371.0088

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        x = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1))
            * cos(radians(lat2))
            * sin(dlon / 2) ** 2
        )

        return 2 * r * atan2(sqrt(x), sqrt(1 - x))

    distances = [0.0]

    for i in range(1, len(mirrored_coordinates)):
        segment = haversine_km(
            mirrored_coordinates[i - 1],
            mirrored_coordinates[i],
        )

        distances.append(distances[-1] + segment)

    return distances


def write_kml_save(
    saved: dict,
    coordinates: list[tuple[float, float]],
    distances_km: list[float],
    output_path: Path,
    source_path: Path,
    requested_loop_distance_km: float,
    selected_distance_km: float,
) -> None:
    """Write the mirrored route as a dashboard-compatible .kml.save."""

    profile = saved.setdefault("profile", {})

    # Dashboard coordinates are stored as [latitude, longitude].
    profile["Coordinates"] = [
        [latitude, longitude]
        for latitude, longitude in coordinates
    ]

    profile["Distance"] = distances_km

    # Keep the original save structure and metadata intact.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Mirrored route generated successfully.")
    print(f"Requested total loop distance: {requested_loop_distance_km:.3f} km")
    print(f"Requested outbound distance: {requested_loop_distance_km / 2:.3f} km")
    print(f"Selected outbound distance: {selected_distance_km:.3f} km")
    print(f"Generated total distance: {distances_km[-1]:.3f} km")
    print(f"Outbound points: {len(coordinates) // 2 + 1}")
    print(f"Total points: {len(coordinates)}")
    print(f"Source: {source_path}")
    print(f"Wrote: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cut a dashboard route at half distance, mirror it, "
            "and create a dashboard .kml.save file."
        )
    )

    parser.add_argument(
        "save_file",
        type=Path,
        help="Input dashboard .kml.save file",
    )

    parser.add_argument(
        "total_loop_distance_km",
        type=float,
        help="Desired full out-and-back loop distance in km",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("mirrored_loop.kml.save"),
        help="Output dashboard .kml.save path",
    )

    arguments = parser.parse_args()

    # Load complete save file.
    saved = load_save(arguments.save_file)

    # Extract original route.
    distances, coordinates = load_profile(saved)

    # Select outbound half.
    outbound, index, selected_distance = route_half_at_distance(
        distances,
        coordinates,
        arguments.total_loop_distance_km,
    )

    # Mirror outbound route.
    mirrored = mirror_route(outbound)

    # Recalculate cumulative distance for the complete mirrored route.
    mirrored_distances = build_mirrored_distances(
        mirrored,
        distances[: len(outbound)],
    )

    # Write dashboard .kml.save.
    write_kml_save(
        saved=saved,
        coordinates=mirrored,
        distances_km=mirrored_distances,
        output_path=arguments.output,
        source_path=arguments.save_file,
        requested_loop_distance_km=arguments.total_loop_distance_km,
        selected_distance_km=selected_distance,
    )


if __name__ == "__main__":
    main()
