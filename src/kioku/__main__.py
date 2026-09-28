from __future__ import annotations

import sys


def main() -> int:
    # Before any Qt import: a stable taskbar identity so Windows shows Kioku's own icon.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Kioku.PhotoSearch")

    from PyQt6.QtWidgets import QApplication

    from . import APP_NAME, __version__
    from .gui import STYLE, MainWindow, app_icon

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
