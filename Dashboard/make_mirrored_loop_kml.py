"""Create a mirrored out-and-back KML from a dashboard .kml.save route.

The dashboard save's profile.Coordinates are (latitude, longitude), whereas
KML coordinates must be written as longitude,latitude,altitude.

Example:
    python make_mirrored_loop_kml.py \
        "Saves/2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop.kml.save" \
        22.4 \
        --output mirrored_rustenburg_loop.kml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def load_profile(save_path: Path) -> tuple[list[float], list[tuple[float, float]]]:
    """Load the distance and (latitude, longitude) arrays from a .kml.save."""
    try:
        saved = json.loads(save_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Save file not found: {save_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Not valid dashboard save JSON: {save_path} ({exc})") from exc

    profile = saved.get("profile") or {}
    distances = profile.get("Distance") or []
    coordinates = profile.get("Coordinates") or []
    if len(distances) != len(coordinates) or len(distances) < 2:
        raise SystemExit(
            "profile.Distance and profile.Coordinates must both exist, have equal "
            "lengths, and contain at least two points."
        )

    try:
        return (
            [float(distance) for distance in distances],
            [(float(lat), float(lon)) for lat, lon in coordinates],
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit("Distances or coordinates contain non-numeric values.") from exc


def route_half_at_distance(
    distances_km: list[float],
    coordinates: list[tuple[float, float]],
    total_loop_distance_km: float,
) -> tuple[list[tuple[float, float]], int, float]:
    """Return coordinates through the sample closest to half the requested loop.

    The selected index is included: it is the turn-around coordinate.
    """
    if total_loop_distance_km <= 0:
        raise ValueError("Total loop distance must be greater than zero.")

    half_distance_km = total_loop_distance_km / 2.0
    closest_index = min(
        range(len(distances_km)),
        key=lambda index: abs(distances_km[index] - half_distance_km),
    )
    return coordinates[: closest_index + 1], closest_index, distances_km[closest_index]


def mirror_route(outbound_coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Mirror an outbound route using route_combiner-style ordered reversal.

    Equivalent to combining an outbound coordinate array with the same array
    in reverse, omitting the duplicate turn-around coordinate.  The output
    begins and ends at the original start point.
    """
    if len(outbound_coordinates) < 2:
        raise ValueError("At least two outbound coordinates are required to mirror a route.")
    return outbound_coordinates + outbound_coordinates[-2::-1]


def write_kml(coordinates: list[tuple[float, float]], output_path: Path, name: str) -> None:
    """Write one LineString KML, swapping (lat, lon) into KML's lon,lat,0."""
    coordinate_text = "\n          ".join(
        f"{longitude:.8f},{latitude:.8f},0" for latitude, longitude in coordinates
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{escape(name)}</name>
    <Style id="mirroredRoute">
      <LineStyle><color>ff00a5ff</color><width>4</width></LineStyle>
    </Style>
    <Placemark>
      <name>{escape(name)}</name>
      <styleUrl>#mirroredRoute</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {coordinate_text}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a dashboard route at half distance, mirror it, and create a KML."
    )
    parser.add_argument("save_file", type=Path, help="Input dashboard .kml.save file")
    parser.add_argument("total_loop_distance_km", type=float, help="Desired full out-and-back loop distance in km")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("mirrored_loop.kml"), help="Output KML path"
    )
    arguments = parser.parse_args()

    distances, coordinates = load_profile(arguments.save_file)
    outbound, index, selected_distance = route_half_at_distance(
        distances, coordinates, arguments.total_loop_distance_km
    )
    mirrored = mirror_route(outbound)
    write_kml(mirrored, arguments.output, f"Mirrored {arguments.save_file.stem}")

    print(f"Requested half distance: {arguments.total_loop_distance_km / 2:.3f} km")
    print(f"Closest saved distance: {selected_distance:.3f} km (index {index})")
    print(f"Outbound points: {len(outbound)}; mirrored KML points: {len(mirrored)}")
    print(f"Wrote: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
