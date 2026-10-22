"""Russian Qt desktop UI; long operations run in one managed worker."""
import logging
from pathlib import Path
import threading

from PySide6.QtCore import QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from app.audio import AudioEngine, audio_devices, stream_format, voices_dir, prepare_voice, record_voice, listen_voice
from app.inference import InferenceEngine, choose_device, devices
from app.models import ModelManager
from app.settings import data_dir, load_settings, save_settings


class Task(QThread):
    progress = Signal(str, object, object)
    ready = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    level = Signal(float)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.stop = threading.Event()

    def run(self):
        try:
            self.completed.emit(self.operation(self))
        except Exception as error:
            logging.exception("Background operation failed")
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeanVC2 Desktop")
        self.resize(780, 870)
        self.settings = load_settings()
        self.manager = ModelManager()
        self.task = None
        self.audio = None
        self.closing = False
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #e8edf5; }
            QMainWindow, QScrollArea, #content { background: #101722; }
            QGroupBox { background: #182333; border: 1px solid #2d3c50; border-radius: 10px;
                        margin-top: 14px; padding: 18px 14px 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 6px; }
            QComboBox, QSpinBox { background: #101b2a; border: 1px solid #43546c;
                                  border-radius: 5px; padding: 7px; min-height: 18px; }
            QComboBox QAbstractItemView { background: #182333; selection-background-color: #286dbe; }
            QPushButton { background: #27394f; border: 1px solid #425773; border-radius: 6px; padding: 9px 14px; }
            QPushButton:hover { background: #355272; }
            QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled { color: #8090a4; }
            QPushButton#start { background: #3478d3; color: white; font-weight: 600; }
            QProgressBar { background: #101b2a; border: 0; border-radius: 4px; text-align: center; min-height: 16px; }
            QProgressBar::chunk { background: #429cec; border-radius: 4px; }
            QSlider::groove:horizontal { height: 5px; background: #344660; }
            QSlider::handle:horizontal { background: #65b1ff; width: 16px; margin: -6px 0; border-radius: 8px; }
        """)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = QLabel("MeanVC2 Desktop")
        title.setStyleSheet("font-size: 26px; font-weight: 650;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Преобразование голоса на вашем компьютере"))
        self.config_groups = []

        model_form = self.group(layout, "Модель")
        self.model = QComboBox()
        self.model.setMinimumContentsLength(25)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for name in self.manager.catalog():
            self.model.addItem(name, name)
        selected = self.settings.get("model")
        if selected and self.model.findData(selected) < 0:
            self.model.addItem(Path(selected).name, selected)
        self.restore(self.model, selected)
        model_form.addRow("Модель", self.model)
        self.profile = QComboBox()
        self.profile.addItem("120 мс + 40 мс — качество", "120ms")
        self.profile.addItem("40 мс + 40 мс — короткий контекст", "40ms")
        self.restore(self.profile, self.settings.get("profile", "120ms"))
        model_form.addRow("Конфигурация", self.profile)
        model_form.addRow(self.row(self.button("Обновить список", self.discover),
                                    self.button("Скачать / обновить", self.download),
                                    self.button("Локальная модель…", self.local_model)))
        self.download_bar = QProgressBar()
        self.download_bar.setRange(0, 100)
        self.download_bar.setValue(0)
        model_form.addRow(self.download_bar)
        self.model_status = QLabel("При первом запуске скачайте модель и вспомогательные файлы.")
        self.model_status.setWordWrap(True)
        model_form.addRow(self.model_status)

        compute_form = self.group(layout, "Вычисления")
        self.compute = QComboBox()
        for label, value in (("Автоматически", "auto"), ("Процессор (CPU)", "cpu"), ("Видеокарта (CUDA)", "cuda")):
            self.compute.addItem(label, value)
        self.restore(self.compute, self.settings.get("compute", "auto"))
        compute_form.addRow("Устройство", self.compute)
        self.gpu = QComboBox()
        for index, name in devices():
            self.gpu.addItem(f"{index}: {name}", index)
        self.restore(self.gpu, self.settings.get("gpu", 0))
        compute_form.addRow("Видеокарта", self.gpu)
        self.compute_info = QLabel()
        self.compute_info.setWordWrap(True)
        compute_form.addRow(self.compute_info)
        self.compute.currentIndexChanged.connect(self.update_compute)
        self.gpu.currentIndexChanged.connect(self.update_compute)
        self.update_compute()

        voice_form = self.group(layout, "Целевой голос")
        self.voice = QComboBox()
        self.refresh_voices(self.settings.get("voice"))
        voice_form.addRow("Образец", self.voice)
        voice_form.addRow(self.row(self.button("Добавить WAV…", self.import_voice),
                                   self.button("Прослушать", self.listen), self.button("Удалить", self.delete_voice)))
        self.record_seconds = QSpinBox()
        self.record_seconds.setRange(1, 30)
        self.record_seconds.setValue(10)
        self.record_seconds.setSuffix(" сек.")
        voice_form.addRow(self.row(self.button("Записать с микрофона", self.record), self.record_seconds))
        voice_form.addRow(QLabel("Чистая речь без музыки, 1–30 секунд. Образцы хранятся локально."))

        audio_form = self.group(layout, "Аудио")
        self.input = QComboBox()
        self.output = QComboBox()
        self.refresh_audio()
        audio_form.addRow("Вход", self.input)
        audio_form.addRow("Выход", self.output)
        audio_form.addRow(self.button("Обновить устройства", lambda: self.refresh_audio(True)))
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setAccessibleName("Уровень входного сигнала")
        audio_form.addRow("Входной уровень", self.level)
        self.rate = QComboBox()
        for label, rate in (("Автоматически (частота устройства)", 0), ("16 000 Гц", 16000),
                            ("44 100 Гц", 44100), ("48 000 Гц", 48000), ("96 000 Гц", 96000)):
            self.rate.addItem(label, rate)
        self.restore(self.rate, self.settings.get("rate", 0))
        audio_form.addRow("Частота устройств", self.rate)

        # Volume remains available while the stream is running.
        volume_row = QHBoxLayout()
        volume_row.addWidget(QLabel("Громкость выхода"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 200)
        self.volume.setAccessibleName("Громкость выхода в процентах")
        self.volume.setValue(max(0, min(200, int(self.settings.get("volume", 100)))))
        self.output_volume = self.volume.value() / 100
        self.volume_label = QLabel(f"{self.volume.value()} %")
        self.volume.valueChanged.connect(self.set_volume)
        volume_row.addWidget(self.volume, 1)
        volume_row.addWidget(self.volume_label)
        layout.addLayout(volume_row)

        advanced = self.group(layout, "Параметры обработки")
        self.steps = QSpinBox()
        self.steps.setRange(1, 8)
        self.steps.setValue(int(self.settings.get("steps", 2)))
        advanced.addRow("Шаги преобразования", self.steps)
        self.threads = QSpinBox()
        self.threads.setRange(1, 32)
        self.threads.setValue(int(self.settings.get("threads", 1)))
        advanced.addRow("Потоки CPU", self.threads)
        self.buffer = QSpinBox()
        self.buffer.setRange(20, 500)
        self.buffer.setSingleStep(20)
        self.buffer.setSuffix(" мс")
        self.buffer.setValue(int(self.settings.get("buffer", 160)))
        advanced.addRow("Буфер воспроизведения", self.buffer)
        hint = QLabel("Больше шагов — выше нагрузка. Больший буфер уменьшает прерывания звука.")
        hint.setWordWrap(True)
        advanced.addRow(hint)

        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(24, 10, 24, 16)
        self.start_button = self.button("Начать преобразование", self.start_conversion)
        self.start_button.setObjectName("start")
        self.stop_button = self.button("Остановить", self.stop)
        self.stop_button.setEnabled(False)
        footer_layout.addWidget(self.row(self.start_button, self.stop_button))
        self.status = QLabel("Готово к работе")
        self.status.setWordWrap(True)
        footer_layout.addWidget(self.status)
        self.metrics = QLabel("Инференс: —   ·   Задержка тракта ≈ —")
        self.metrics.setWordWrap(True)
        footer_layout.addWidget(self.metrics)
        self.errors = QLabel()
        self.errors.setStyleSheet("color: #ffadad;")
        self.errors.setWordWrap(True)
        self.errors.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer_layout.addWidget(self.errors)
        footer_layout.addWidget(self.row(self.button("Открыть журнал", lambda: self.open_path(data_dir() / "app.log")),
                                   self.button("Папка приложения", lambda: self.open_path(data_dir()))))
        layout.addStretch()
        scroll.setWidget(content)
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(scroll, 1)
        central_layout.addWidget(footer)
        self.setCentralWidget(central)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(100)

    def group(self, layout, name):
        group = QGroupBox(name)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addWidget(group)
        self.config_groups.append(group)
        return form

    @staticmethod
    def row(*widgets):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for item in widgets:
            layout.addWidget(item)
        return widget

    @staticmethod
    def button(text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def restore(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def update_compute(self):
        self.gpu.setEnabled(self.compute.currentData() != "cpu" and self.gpu.count() > 0)
        try:
            device = choose_device(self.compute.currentData(), self.gpu.currentData() or 0)
            self.compute_info.setText("Используется: " + ("Процессор (CPU)" if device == "cpu" else self.gpu.currentText()))
        except ValueError as error:
            self.compute_info.setText(str(error))

    def refresh_audio(self, refresh=False):
        selected_in = self.input.currentData() or self.settings.get("input")
        selected_out = self.output.currentData() or self.settings.get("output")
        try:
            self.device_list = audio_devices(refresh)
            for combo, kind, selected in ((self.input, "inputs", selected_in), (self.output, "outputs", selected_out)):
                combo.clear()
                combo.addItem("Системное устройство по умолчанию", None)
                for device in self.device_list:
                    if device[kind]:
                        combo.addItem(device["key"], device["key"])
                self.restore(combo, selected)
        except Exception as error:
            logging.exception("Audio enumeration failed")
            self.device_list = []
            if hasattr(self, "errors"):
                self.errors.setText(str(error))

    def device_index(self, combo):
        key = combo.currentData()
        if key is None:
            return None
        return next(device["index"] for device in self.device_list if device["key"] == key)

    def refresh_voices(self, selection=None):
        self.voice.clear()
        for path in sorted(voices_dir().glob("*.wav")):
            self.voice.addItem(path.stem, str(path))
        self.restore(self.voice, str(selection) if selection else None)

    def selected_voice(self):
        path = self.voice.currentData()
        if not path or not Path(path).is_file():
            raise ValueError("Добавьте или запишите образец целевого голоса")
        return path

    def run_task(self, text, operation, completed=lambda result: None):
        if self.task:
            return
        self.errors.clear()
        self.status.setText(text)
        self.persist()
        for group in self.config_groups:
            group.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.task = Task(operation, self)
        self.task.completed.connect(completed)
        self.task.failed.connect(self.show_error)
        self.task.progress.connect(self.show_progress)
        self.task.ready.connect(self.status.setText)
        self.task.level.connect(lambda level: self.level.setValue(min(100, round(level * 200))))
        self.task.finished.connect(self.finished)
        self.task.start()

    def show_progress(self, name, current, total):
        self.model_status.setText(f"{name}: {current / 1024**2:.1f} / {total / 1024**2:.1f} МБ")
        self.download_bar.setRange(0, 100 if total else 0)
        if total:
            self.download_bar.setValue(round(100 * current / total))

    def show_error(self, message):
        self.errors.setText(message)
        self.status.setText("Операция завершилась с ошибкой. Подробности — в журнале.")

    def finished(self):
        task, self.task = self.task, None
        task.deleteLater()
        self.audio = None
        for group in self.config_groups:
            group.setEnabled(True)
        self.update_compute()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.level.setValue(0)
        self.download_bar.setRange(0, 100)
        if not self.errors.text():
            self.status.setText("Готово к работе")
        self.persist()
        if self.closing:
            self.close()

    def stop(self):
        if self.task:
            self.task.stop.set()
            self.status.setText("Остановка… Дождитесь завершения текущей операции.")
            self.stop_button.setEnabled(False)

    def discover(self):
        def completed(names):
            selected = self.model.currentData()
            self.model.clear()
            for name in names:
                self.model.addItem(name, name)
            if selected and Path(selected).is_absolute():
                self.model.addItem(Path(selected).name, selected)
            self.restore(self.model, selected)
            self.model_status.setText(f"Доступно моделей: {len(names)}")
        self.run_task("Получение списка моделей…", lambda task: self.manager.discover(), completed)

    def local_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "", "Safetensors (*.safetensors)")
        if path:
            if self.model.findData(path) < 0:
                self.model.addItem(Path(path).name, path)
            self.restore(self.model, path)
            self.persist()

    def download(self):
        selection, profile = self.model.currentData(), self.profile.currentData()
        def operation(task):
            def progress(name, current, total):
                if task.stop.is_set():
                    raise RuntimeError("Скачивание остановлено; кеш сохранён для повторного запуска")
                task.progress.emit(name, current, total)
            return self.manager.download(selection, profile, progress)
        self.run_task("Скачивание модели и вспомогательных файлов…", operation,
                      lambda path: (self.model_status.setText("Модель и вспомогательные файлы сохранены. Можно работать без интернета."),
                                    self.download_bar.setValue(100)))

    def import_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, "Добавить целевой голос", "", "WAV (*.wav)")
        if path:
            self.run_task("Подготовка образца…", lambda task: prepare_voice(path), self.refresh_voices)

    def record(self):
        device, seconds = self.device_index(self.input), self.record_seconds.value()
        self.run_task(f"Запись образца: до {seconds} секунд…",
                      lambda task: record_voice(device, seconds, task.stop, task.level.emit), self.refresh_voices)

    def listen(self):
        try:
            path, device = self.selected_voice(), self.device_index(self.output)
            self.run_task("Прослушивание…", lambda task: listen_voice(path, device, task.stop))
        except Exception as error:
            self.show_error(str(error))

    def delete_voice(self):
        try:
            path = Path(self.selected_voice())
            if QMessageBox.question(self, "Удаление образца", f"Удалить «{path.stem}»?") == QMessageBox.StandardButton.Yes:
                path.unlink()
                self.refresh_voices()
                self.persist()
        except Exception as error:
            self.show_error(str(error))

    def start_conversion(self):
        try:
            reference = self.selected_voice()
            options = dict(model=self.model.currentData(), reference=reference,
                           mode=self.compute.currentData(), gpu=self.gpu.currentData() or 0,
                           profile=self.profile.currentData(), steps=self.steps.value(), threads=self.threads.value())
            choose_device(options["mode"], options["gpu"])
            input_device, output_device = self.device_index(self.input), self.device_index(self.output)
            rate, buffer_ms = self.rate.currentData(), self.buffer.value()
            def operation(task):
                stream_format(input_device, "input", rate)
                stream_format(output_device, "output", rate)
                engine = InferenceEngine(**options, manager=self.manager)
                if task.stop.is_set():
                    return
                self.audio = AudioEngine(engine, input_device, output_device, rate,
                                         self.output_volume, buffer_ms)
                self.audio.run(task.stop, lambda: task.ready.emit("Преобразование работает · " + engine.device))
            self.run_task("Загрузка моделей и подготовка голоса…", operation)
        except Exception as error:
            logging.exception("Cannot start conversion")
            self.show_error(str(error))

    def set_volume(self, value):
        self.output_volume = value / 100
        self.volume_label.setText(f"{value} %")
        if self.audio:
            self.audio.volume = value / 100

    def update_metrics(self):
        if self.audio:
            stats = self.audio.stats
            self.level.setValue(min(100, round(stats.level * 200)))
            self.metrics.setText(f"Инференс: {stats.processing_ms:.1f} мс / 160 мс · "
                                 f"Задержка тракта ≈ {stats.latency_ms:.0f} мс · "
                                 f"Прерывания входа/выхода: {stats.input_overflows}/{stats.output_underflows}")

    def persist(self):
        try:
            save_settings(dict(model=self.model.currentData(), profile=self.profile.currentData(),
                compute=self.compute.currentData(), gpu=self.gpu.currentData(), voice=self.voice.currentData(),
                input=self.input.currentData(), output=self.output.currentData(), rate=self.rate.currentData(),
                volume=self.volume.value(), steps=self.steps.value(), threads=self.threads.value(), buffer=self.buffer.value()))
        except OSError as error:
            logging.exception("Cannot save settings")
            self.errors.setText("Не удалось сохранить настройки: " + str(error))

    @staticmethod
    def open_path(path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event: QCloseEvent):
        if self.task:
            self.closing = True
            self.stop()
            event.ignore()
            return
        self.persist()
        event.accept()
