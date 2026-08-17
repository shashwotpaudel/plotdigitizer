"""Desktop application entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

__all__ = ["main"]

# A dark palette, so a figure on a white background is the brightest thing on screen.
STYLE_SHEET = """
QMainWindow, QDockWidget, QWidget { background: #23272e; color: #e9ecef; }
QToolBar { background: #1e2126; border: none; padding: 4px; spacing: 2px; }
QToolBar QToolButton { padding: 5px 10px; border-radius: 4px; }
QToolBar QToolButton:hover { background: #343a40; }
QToolBar QToolButton:checked { background: #364fc7; }
QTabWidget::pane { border: 1px solid #343a40; }
QTabBar::tab { background: #2b3038; padding: 7px 12px; border: 1px solid #343a40;
               border-bottom: none; }
QTabBar::tab:selected { background: #364fc7; }
QGroupBox { border: 1px solid #343a40; border-radius: 4px; margin-top: 9px;
            padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px;
                   color: #adb5bd; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTableWidget, QListWidget {
    background: #1b1e23; border: 1px solid #343a40; border-radius: 3px; padding: 3px; }
QPushButton { background: #343a40; border: 1px solid #495057; border-radius: 4px;
              padding: 6px 12px; }
QPushButton:hover { background: #3f464e; }
QPushButton:default { background: #364fc7; border-color: #4c6ef5; }
QHeaderView::section { background: #2b3038; border: none; padding: 4px; }
QStatusBar { background: #1e2126; }
QStatusBar QLabel { padding: 0 8px; }
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plotdigitizer-gui",
                                     description="Intelligent plot digitizer (desktop app)")
    parser.add_argument("image", nargs="?", help="image to open on startup")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="compute device (default: auto)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Plot Digitizer")
    app.setOrganizationName("plotdigitizer")
    app.setStyleSheet(STYLE_SHEET)

    window = MainWindow(device=args.device)
    window.show()

    if args.image:
        path = Path(args.image)
        if path.exists():
            window.load_image(path)
        else:
            logging.error("%s does not exist", path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
