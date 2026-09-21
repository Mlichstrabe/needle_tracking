"""Independent STL/archive checks for the T11 slider-sleeve delivery.

This verifier intentionally does not import CadQuery, so it can still report
the final mesh and ZIP status when the native CAD library exits abnormally.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "cad_laser100m_T11_slider_sleeve"
BUNDLE = ROOT / "output" / "T11_SLIDER_SLEEVE_STEP_STL_PACKAGE.zip"


def verify_mesh(path: Path):
    mesh = trimesh.load_mesh(path, process=True)
    components = len(mesh.split(only_watertight=False))
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(components),
        "bounds_mm": mesh.bounds.tolist(),
        "faces": int(len(mesh.faces)),
    }


def main():
    names = ["T11_SLIDER_SLEEVE_FINISHED.stl", "T11_SLIDER_SLEEVE_PRINT_WITH_FOOT.stl"]
    stls = {name: verify_mesh(OUT / name) for name in names}
    for name, row in stls.items():
        if not row["watertight"] or not row["winding_consistent"] or row["components"] != 1:
            raise SystemExit(f"STL check failed: {name}: {row}")
    with zipfile.ZipFile(BUNDLE) as z:
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"ZIP CRC check failed: {bad}")
        archive_entries = len(z.infolist())
    checksums = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in OUT.iterdir()
        if f.is_file() and f.name not in {"sha256.json", "independent_validation.json"}
    }
    recorded = json.loads((OUT / "sha256.json").read_text(encoding="utf-8"))
    if checksums != recorded:
        raise SystemExit("sha256.json does not match generated files")
    result = {"independent_validation": "PASS", "stls": stls,
              "archive_entries": archive_entries, "archive_integrity": "PASS",
              "sha256_manifest": "PASS"}
    (OUT / "independent_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
