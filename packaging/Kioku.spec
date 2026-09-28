# PyInstaller spec: onedir GUI build of Kioku, with the default model (fetch_model.DEFAULT) bundled.
#
#   .venv\Scripts\python.exe packaging\build.py            (runs this, then Inno Setup)
#
# Output: dist\Kioku\Kioku.exe. No UPX (it trips antivirus and breaks some Qt DLLs).

import hashlib
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
# The build ships only the default model's files, and only if they match their pins in
# fetch_model.py.
_spec = importlib.util.spec_from_file_location("fetch_model", ROOT / "packaging" / "fetch_model.py")
fetch_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_model)
MODEL_NAME = fetch_model.DEFAULT
MODEL = ROOT / "models" / MODEL_NAME
MODEL_FILES = fetch_model.MODELS[MODEL_NAME][2]
for _repo_path, name, expected in MODEL_FILES:
    f = MODEL / name
    if not f.is_file() or fetch_model.sha256(f) != expected:
        raise SystemExit(f"{f} is missing or does not match its pin: run packaging\\fetch_model.py")

ENTRY = Path(workpath) / "kioku_entry.py"
ENTRY.parent.mkdir(parents=True, exist_ok=True)
ENTRY.write_text("import sys\nfrom kioku.__main__ import main\nsys.exit(main())\n", encoding="utf-8")

RESOURCES = SRC / "kioku" / "resources"
DATAS = [(str(MODEL / name), f"models/{MODEL_NAME}") for _r, name, _h in MODEL_FILES]
DATAS += [(str(p), "kioku/resources") for p in RESOURCES.iterdir() if p.is_file()]
DATAS += [(str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_NOTICES.md"), ".")]

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=collect_dynamic_libs("onnxruntime"),  # onnxruntime_providers + DirectML.dll
    datas=DATAS,
    hiddenimports=["kioku.gui"],
    excludes=["tkinter", "pytest", "pytestqt", "matplotlib", "scipy", "pandas", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Kioku",
    console=False,
    icon=str(RESOURCES / "app.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="Kioku")
