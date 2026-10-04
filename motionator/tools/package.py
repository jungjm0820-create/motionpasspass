"""Assemble the Blender extension zip: addon/*.py + wheel(s) + generated manifest.

Usage (CI):    python tools/package.py
Local test:    python tools/package.py --pattern "*cp312*linux*.whl" --wheelhouse /tmp/wh
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--wheelhouse", default=str(ROOT / "wheelhouse"))
ap.add_argument("--pattern", default="*cp313*win_amd64.whl")
ap.add_argument("--out", default=str(ROOT / "dist"))
args = ap.parse_args()

wheels = sorted(glob.glob(os.path.join(args.wheelhouse, args.pattern)))
if not wheels:
    sys.exit(f"No wheel matching {args.pattern} in {args.wheelhouse}")

stage = ROOT / "build" / "motionator"
if stage.parent.exists():
    shutil.rmtree(stage.parent)
(stage / "wheels").mkdir(parents=True)

for py in (ROOT / "addon").glob("*.py"):
    shutil.copy(py, stage / py.name)
names = []
for w in wheels:
    shutil.copy(w, stage / "wheels" / os.path.basename(w))
    names.append(f'"./wheels/{os.path.basename(w)}"')

manifest = (ROOT / "addon" / "blender_manifest.toml.in").read_text(encoding="utf-8")
manifest = manifest.replace("@WHEELS@", ", ".join(names))
(stage / "blender_manifest.toml").write_text(manifest, encoding="utf-8")

try:
    import tomllib
    tomllib.loads(manifest)
except ImportError:
    pass

out = Path(args.out)
out.mkdir(exist_ok=True)
zpath = out / "motionator-0.1.0-windows_x64.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(stage.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(stage).as_posix())
print("Built", zpath)
for n in zipfile.ZipFile(zpath).namelist():
    print("  ", n)
