"""Build dist\\Kioku\\ (PyInstaller) and then dist\\Kioku-Setup-<version>.exe (Inno Setup 6).

    .venv\\Scripts\\python.exe packaging\\build.py            # both
    .venv\\Scripts\\python.exe packaging\\build.py --no-installer
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from kioku import __version__  # noqa: E402


def find_iscc() -> Path:
    candidates = [os.environ.get("ISCC"), shutil.which("ISCC")]
    for base in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        root = os.environ.get(base)
        if root:
            sub = ("Programs", "Inno Setup 6") if base == "LOCALAPPDATA" else ("Inno Setup 6",)
            candidates.append(str(Path(root, *sub, "ISCC.exe")))
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise SystemExit("Inno Setup 6 (ISCC.exe) not found; install it or set ISCC")


def main() -> int:
    installer = "--no-installer" not in sys.argv
    iscc = find_iscc() if installer else None  # fail fast, before the long freeze
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
         str(HERE / "Kioku.spec")],
        check=True, cwd=ROOT,
    )  # fmt: skip
    exe = ROOT / "dist" / "Kioku" / "Kioku.exe"
    if not exe.is_file():
        raise SystemExit(f"PyInstaller produced no {exe}")
    print("built", exe)
    if iscc:
        subprocess.run([str(iscc), "/Q", f"/DAppVersion={__version__}", str(HERE / "installer.iss")], check=True)
        setup = ROOT / "dist" / f"Kioku-Setup-{__version__}.exe"
        print("built", setup, f"({setup.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
