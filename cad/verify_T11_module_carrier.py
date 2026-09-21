"""Independent STL/archive checks for the T11 module-carrier delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "cad_laser100m_T11_module_carrier"
BUNDLE = ROOT / "output" / "T11_MODULE_CARRIER_STEP_STL_PACKAGE.zip"


def check_mesh(path: Path):
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
    meshes = {}
    for path in sorted(OUT.glob("*.stl")):
        row = check_mesh(path)
        if not row["watertight"] or not row["winding_consistent"] or row["components"] != 1:
            raise SystemExit(f"STL check failed: {path.name}: {row}")
        meshes[path.name] = row
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
    result = {"independent_validation": "PASS", "stls": meshes,
              "archive_entries": archive_entries, "archive_integrity": "PASS",
              "sha256_manifest": "PASS"}
    (OUT / "independent_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"independent_validation": "PASS", "stls": len(meshes),
                      "archive_entries": archive_entries, "archive_integrity": "PASS",
                      "sha256_manifest": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
