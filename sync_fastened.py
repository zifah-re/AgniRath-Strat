"""
sync_fastened.py — keep Model_fastened_hafiz in sync with Model.

The two folders are the SAME codebase; the ONLY intended difference is
``configs/solver_config.py`` (the fastened build flips 4 performance flags).
So whenever you change something in Model/, run this to copy everything across
EXCEPT that one config file — the fastened flags are preserved automatically.

Run it from the folder that contains both Model/ and Model_fastened_hafiz/:

    python sync_fastened.py            # copy Model -> Model_fastened_hafiz
    python sync_fastened.py --check    # just report what differs, copy nothing

What it does / doesn't touch:
  * copies every .py and data-less source file from Model/ into
    Model_fastened_hafiz/, overwriting;
  * SKIPS configs/solver_config.py (so your fastened flags stay put);
  * SKIPS data/ and output/ (large, and each folder points at its own);
  * SKIPS caches (__pycache__, .pytest_cache, *.pyc).
"""
from __future__ import annotations
import filecmp
import os
import shutil
import sys

SRC = "Model"
DST = "Model_fastened_hafiz"
SKIP_FILES = {os.path.join("configs", "solver_config.py")}
SKIP_DIRS = {"data", "output", "__pycache__", ".pytest_cache"}


def _iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if rel in SKIP_FILES:
                continue
            yield rel


def main() -> None:
    check = "--check" in sys.argv[1:]
    if not (os.path.isdir(SRC) and os.path.isdir(DST)):
        raise SystemExit(
            f"Run this from the folder containing both {SRC}/ and {DST}/ "
            f"(cwd={os.getcwd()}).")

    copied, same, missing = 0, 0, 0
    for rel in _iter_files(SRC):
        s = os.path.join(SRC, rel)
        d = os.path.join(DST, rel)
        if os.path.exists(d) and filecmp.cmp(s, d, shallow=False):
            same += 1
            continue
        if not os.path.exists(d):
            missing += 1
        if check:
            print(("NEW " if not os.path.exists(d) else "DIFF"), rel)
            continue
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        shutil.copy2(s, d)
        copied += 1

    # Sanity: confirm the intended single difference is intact.
    cfg_rel = os.path.join("configs", "solver_config.py")
    cfg_differs = not filecmp.cmp(os.path.join(SRC, cfg_rel),
                                  os.path.join(DST, cfg_rel), shallow=False)

    if check:
        print(f"\n{same} identical, {missing} missing/diff (not counting "
              f"the intentionally-skipped {cfg_rel}).")
    else:
        print(f"Synced: {copied} file(s) copied, {same} already identical.")
    print(f"configs/solver_config.py intentionally NOT synced — "
          f"fastened flags {'preserved (good)' if cfg_differs else 'MATCH BASELINE — did you flip them?'}.")


if __name__ == "__main__":
    main()
