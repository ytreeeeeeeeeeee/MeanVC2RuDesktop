"""Application entry point, also used by the frozen executable."""
import argparse
import logging
from pathlib import Path
import sys

from app.settings import configure_logging


def main():
    parser = argparse.ArgumentParser(description="MeanVC2 Desktop")
    parser.add_argument("--diagnostics", type=Path, help="Write environment diagnostics as JSON")
    parser.add_argument("--smoke-test", type=Path, help="Open GUI, save screenshot and exit")
    parser.add_argument("--benchmark", type=Path, help="Reference WAV for streaming inference benchmark")
    parser.add_argument("--source", type=Path, help="Source WAV for benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="meanvc2_ru.safetensors")
    parser.add_argument("--profile", choices=["120ms", "40ms"], default="120ms")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--audio-test", action="store_true", help="Include physical audio I/O in benchmark")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()
    configure_logging()
    if args.diagnostics or args.benchmark:
        from app.checks import diagnostics, benchmark
        try:
            if args.diagnostics:
                diagnostics(args.diagnostics)
            else:
                benchmark(args)
            return 0
        except Exception:
            logging.exception("Command failed")
            raise
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from app.gui import MainWindow
    application = QApplication(sys.argv[:1])
    application.setApplicationName("MeanVC2 Desktop")
    application.setOrganizationName("MeanVC2Desktop")
    def exception_hook(kind, value, traceback):
        logging.error("Unhandled exception", exc_info=(kind, value, traceback))
        QMessageBox.critical(None, "Ошибка MeanVC2 Desktop", str(value))
    sys.excepthook = exception_hook
    window = MainWindow()
    window.show()
    if args.smoke_test:
        def capture():
            args.smoke_test.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.smoke_test.resolve()))
            window.close()
        QTimer.singleShot(1500, capture)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
