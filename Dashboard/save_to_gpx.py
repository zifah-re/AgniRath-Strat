#!/usr/bin/env python3
"""
Convert Sasol Solar Challenge route ".save" files (JSON exports from a
Jupyter/folium notebook cell) into standard .gpx and .kml files that any
GPS app (OsmAnd, Organic Maps, Maps.me, QGIS, Google Earth, etc.) can open.

The .save file isn't real KML -- it's a saved notebook cell containing a
folium map render, but it carries the real route data inside a "profile"
key: Coordinates, Altitude, Distance, SpeedLimit, Headings, etc.
This script pulls Coordinates (and Altitude, if present) back out and
writes them as a proper GPX track (with elevation) and a plain KML
LineString.

Default output location:
    If the input is at .../Dashboard/Saves/<name>.kml.save
    outputs go to  .../Dashboard/gpx/<name>.gpx and <name>.kml
    (i.e. a "gpx" folder that is a sibling of the "Saves" folder).
    Override with --outdir <path> if you want somewhere else.

Usage:
    python save_to_gpx.py "Saves\\some route.kml.save"
    python save_to_gpx.py "Saves\\a.kml.save" "Saves\\b.kml.save"
    python save_to_gpx.py --outdir "C:\\path\\to\\gpx" "Saves\\a.kml.save"
"""

import argparse
import json
import sys
from pathlib import Path


def load_route(save_path: Path):
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = data.get("profile", {})
    coords = profile.get("Coordinates")
    if not coords:
        raise ValueError(f"No Coordinates found in {save_path.name}")

    altitudes = profile.get("Altitude")
    if altitudes and len(altitudes) != len(coords):
        altitudes = None  # mismatched length, safer to drop

    name = data.get("placemark_name") or save_path.stem
    return name, coords, altitudes


def clean_stem(save_path: Path) -> str:
    """
    Strip the .save suffix, and also strip a trailing .kml if present,
    so "route.kml.save" -> "route" instead of "route.kml".
    """
    stem = save_path.stem  # removes ".save"
    if stem.lower().endswith(".kml"):
        stem = stem[: -len(".kml")]
    return stem


def default_output_dir(save_path: Path) -> Path:
    """
    If the file lives in .../Dashboard/Saves/x.save, put outputs in
    .../Dashboard/gpx/. Falls back to a "gpx" folder next to the input
    file if there's no obvious Saves-style parent.
    """
    parent = save_path.resolve().parent
    if parent.name.lower() == "saves":
        return parent.parent / "gpx"
    return parent / "gpx"


def write_gpx(out_path: Path, name: str, coords, altitudes=None):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="SasolSolarChallengeExport" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk>",
        f"    <name>{name}</name>",
        "    <trkseg>",
    ]
    for i, (lat, lon) in enumerate(coords):
        if altitudes:
            lines.append(
                f'      <trkpt lat="{lat}" lon="{lon}">'
                f"<ele>{altitudes[i]}</ele></trkpt>"
            )
        else:
            lines.append(f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>')
    lines += ["    </trkseg>", "  </trk>", "</gpx>", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_kml(out_path: Path, name: str, coords, altitudes=None):
    if altitudes:
        coord_str = " ".join(
            f"{lon},{lat},{alt}" for (lat, lon), alt in zip(coords, altitudes)
        )
    else:
        coord_str = " ".join(f"{lon},{lat},0" for lat, lon in coords)

    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Placemark>
      <name>{name}</name>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coord_str}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
'''
    out_path.write_text(kml, encoding="utf-8")


def convert(save_path: Path, outdir: Path):
    name, coords, altitudes = load_route(save_path)
    stem = clean_stem(save_path)

    outdir.mkdir(parents=True, exist_ok=True)
    gpx_path = outdir / f"{stem}.gpx"
    kml_path = outdir / f"{stem}.kml"

    write_gpx(gpx_path, name, coords, altitudes)
    write_kml(kml_path, name, coords, altitudes)
    print(f"{save_path.name}: {len(coords)} points -> {gpx_path} , {kml_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="One or more .save files to convert")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output folder (default: <Dashboard>/gpx, i.e. sibling of Saves/)",
    )
    args = parser.parse_args()

    for arg in args.files:
        p = Path(arg)
        if not p.exists():
            print(f"Skipping {arg}: not found")
            continue
        outdir = Path(args.outdir) if args.outdir else default_output_dir(p)
        try:
            convert(p, outdir)
        except Exception as e:
            print(f"Failed on {p.name}: {e}")


if __name__ == "__main__":
    main()