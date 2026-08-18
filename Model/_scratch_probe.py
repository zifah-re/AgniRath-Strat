"""Scratch probe (deleted after use): measure route kms + KML (False) zones."""
import os, glob, json
import xml.etree.ElementTree as ET

root = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root, "data", "processed")
kml_dir = os.path.join(root, "data", "shaded")


def route_kms(files):
    files = sorted(files)
    tot = 0.0
    stage1 = stage2 = 0.0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        prof = data["profile"]
        d = max(prof["Distance"]) * 1000.0
        bn = os.path.basename(fp).lower()
        if "stage 1" in bn:
            stage1 += d
        elif "stage 2" in bn:
            stage2 += d
        tot += d
    return tot / 1000.0, stage1 / 1000.0, stage2 / 1000.0


for d in range(1, 9):
    files = glob.glob(os.path.join(save_dir, f"*Day {d}*.save"))
    if d == 3:
        for key in ("prahlad", "aryaman"):
            vf = [f for f in files if key in os.path.basename(f).lower()]
            tot, s1, s2 = route_kms(vf)
            print(f"Day {d} [{key}]: total={tot:.2f} km, stage1={s1:.2f}, stage2={s2:.2f} files={len(vf)}")
        continue
    tot, s1, s2 = route_kms(files)
    print(f"Day {d}: total={tot:.2f} km, stage1={s1:.2f}, stage2={s2:.2f} files={len(files)}")


def kml_false_zones(path):
    if not path:
        return []
    tree = ET.parse(path)
    root_el = tree.getroot()
    for el in root_el.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    out = []
    for pm in root_el.findall(".//Placemark"):
        name = pm.find("name")
        if name is not None and "(False)" in (name.text or ""):
            ls = pm.find("LineString")
            n = 0
            if ls is not None and ls.find("coordinates") is not None:
                n = len(ls.find("coordinates").text.strip().split())
            out.append((name.text, n))
    return out


for d in range(1, 9):
    files = sorted(glob.glob(os.path.join(kml_dir, f"*Day {d}*.kml")))
    for f in files:
        print(f"KML Day {d} {os.path.basename(f)}: {kml_false_zones(f)}")
