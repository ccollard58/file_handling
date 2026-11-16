"""PyQt6 GUI for selecting Ollama and hosted vision models, running the evaluation,
and viewing progress and results."""

import os
import sys
from typing import Callable, List

import requests
from PyQt6.QtCore import QObject, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QSpinBox, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ollama_cloud_models import (
    fetch_free_cloud_vision_models,
    fetch_local_vision_models,
)
from openrouter_models import fetch_vision_models as fetch_openrouter_models
from tokenharbor_models import fetch_vision_models as fetch_tokenharbor_models
from nvidia_models import fetch_vision_models as fetch_nvidia_models

RESULT_COLUMNS = [
    "Rank", "Model", "Composite", "Char Acc", "Semantic", "WER",
    "Levenshtein", "Time (s)", "Error",
]

PROVIDER_LABELS = {
    "ollama": "Ollama", "openrouter": "OpenRouter", "tokenharbor": "Token Harbor",
    "nvidia": "NVIDIA", "gemini": "Gemini",
}
PAID_CAPABLE = ("openrouter", "tokenharbor")
DEFAULT_NUM_CTX = 8192
DEFAULT_PDF_DPI = 200
DEFAULT_JUDGES = {
    "ollama": "gpt-oss:120b",
    "openrouter": "openai/gpt-oss-120b:free",
    "tokenharbor": "mimo-v2.6-flash:free",
    "nvidia": "z-ai/glm5",
    "gemini": "gemini-2.5-flash",
}
DEFAULTS = {
    "pdf": "", "truth": "", "output_dir": "evaluation_results",
    "ollama_url": "http://localhost:11434", "provider": "ollama",
    "include_paid": True, "request_timeout": 300,
    "num_ctx": DEFAULT_NUM_CTX, "pdf_dpi": DEFAULT_PDF_DPI,
}


def new_settings() -> QSettings:
    return QSettings("file_handling", "VisionEvaluationGUI")


def resolve_setting(settings: QSettings, key: str, cli_value, default):
    """A non-default command-line value wins; otherwise use the saved value."""
    if cli_value not in (None, default):
        return cli_value
    saved = settings.value(key, default)
    if isinstance(default, bool):
        return saved if isinstance(saved, bool) else str(saved).lower() == "true"
    return saved if saved not in (None, "") else default


class TaskWorker(QObject):
    """Runs a callable on a thread; the callable receives this worker for signals."""

    progress = pyqtSignal(int, int, str)
    debug_logs_ready = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task: Callable[["TaskWorker"], object]):
        super().__init__()
        self._task = task
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            self.finished.emit(self._task(self))
        except Exception as exc:  # surfaced to the UI
            self.failed.emit(str(exc))


def start_worker(owner: QObject, task: Callable[[TaskWorker], object]):
    thread = QThread(owner)
    worker = TaskWorker(task)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    return thread, worker


def make_evaluation_task(pdf, truth, out_dir, url, judge, models: List[str],
                         provider: str = "ollama", api_key: str = "",
                         request_timeout: int = 300,
                         num_ctx: int = DEFAULT_NUM_CTX,
                         pdf_dpi: int = DEFAULT_PDF_DPI,
                         gemini_rate_limits=None):
    def task(worker: TaskWorker):
        from evaluate_handwriting_vision_models import VisionModelEvaluator

        evaluator = VisionModelEvaluator(
            ollama_base_url=url, judge_model=judge, provider=provider,
            request_timeout=request_timeout,
            default_num_ctx=num_ctx, pdf_dpi=pdf_dpi,
            semantic_timeout=request_timeout,
            ollama_api_key=api_key or None if provider == "ollama" else None,
            openrouter_api_key=api_key or None if provider == "openrouter" else None,
            tokenharbor_api_key=api_key or None if provider == "tokenharbor" else None,
            nvidia_api_key=api_key or None if provider == "nvidia" else None,
            gemini_api_key=api_key or None if provider == "gemini" else None,
            gemini_rate_limits=gemini_rate_limits,
        )
        worker.debug_logs_ready.emit(
            evaluator._vision_log_path,
            evaluator._judge_log_path,
        )
        return evaluator.evaluate_all_models(
            pdf, truth, out_dir, models=models,
            progress_callback=worker.progress.emit,
            should_cancel=worker.is_cancelled,
        )
    return task


def make_fetch_task(
    ollama_url: str,
    provider: str = "ollama",
    api_key: str = "",
    include_paid: bool = False,
):
    if provider == "openrouter":
        fetch_vision = lambda: fetch_openrouter_models(
            api_key=api_key or None, include_paid=include_paid
        )
        from openrouter_models import fetch_free_text_models
        fetch_judges = lambda: fetch_free_text_models(api_key=api_key or None)
    elif provider == "tokenharbor":
        fetch_vision = lambda: fetch_tokenharbor_models(
            api_key=api_key or None, include_paid=include_paid
        )
        from tokenharbor_models import fetch_free_text_models
        fetch_judges = lambda: fetch_free_text_models(api_key=api_key or None)
    elif provider == "nvidia":
        fetch_vision = lambda: fetch_nvidia_models(api_key=api_key or None)
        from nvidia_models import fetch_text_models
        fetch_judges = lambda: fetch_text_models(api_key=api_key or None)
    elif provider == "gemini":
        from gemini_models import fetch_text_models, fetch_vision_models
        fetch_vision = lambda: fetch_vision_models(api_key=api_key or None)
        fetch_judges = lambda: fetch_text_models(api_key=api_key or None)
    else:
        from ollama_cloud_models import fetch_text_models

        def fetch_vision():
            local_models = fetch_local_vision_models(
                ollama_url, api_key=api_key or None
            )
            try:
                cloud_models = fetch_free_cloud_vision_models(
                    ollama_url, api_key=api_key or None
                )
            except requests.RequestException as exc:
                return local_models, f"Could not refresh Ollama cloud models: {exc}"
            return list(dict.fromkeys(cloud_models + local_models)), None

        def fetch_judges():
            text_models = fetch_text_models(
                ollama_url, api_key=api_key or None
            )
            local_vision_models = fetch_local_vision_models(
                ollama_url, api_key=api_key or None
            )
            return list(dict.fromkeys(text_models + local_vision_models))

    def fetch(_worker):
        vision_error = None
        if provider == "ollama":
            vision_models, vision_error = fetch_vision()
        else:
            vision_models = fetch_vision()
        try:
            judge_models = fetch_judges()
            judge_error = None
        except Exception as exc:
            judge_models = []
            judge_error = str(exc)
        return vision_models, judge_models, judge_error, vision_error

    return fetch


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Vision Model Evaluation")
        self.resize(1100, 800)
        self._thread = None
        self._worker = None
        self._running = set()
        self._fetch_id = 0

        self._settings = new_settings()
        self._gemini_rate_limits = getattr(args, "gemini_rate_limits", None)

        def setting(key, arg_name, default):
            return resolve_setting(self._settings, key, getattr(args, arg_name, None), default)

        provider = setting("provider", "provider", DEFAULTS["provider"])
        if provider not in DEFAULT_JUDGES:
            provider = DEFAULTS["provider"]
        self._provider = provider

        self.pdf_edit = QLineEdit(setting("pdf", "pdf_path", DEFAULTS["pdf"]))
        self.truth_edit = QLineEdit(setting("truth", "ground_truth_path", DEFAULTS["truth"]))
        self.out_edit = QLineEdit(setting("output_dir", "output_dir", DEFAULTS["output_dir"]))
        self.url_edit = QLineEdit(setting("ollama_url", "ollama_url", DEFAULTS["ollama_url"]))
        self.request_timeout_spin = QSpinBox()
        self.request_timeout_spin.setRange(10, 3600)
        self.request_timeout_spin.setSingleStep(30)
        self.request_timeout_spin.setSuffix(" s")
        self.request_timeout_spin.setToolTip(
            "Maximum wait for each model extraction and judge request, in seconds."
        )
        self.request_timeout_spin.setValue(
            int(setting("request_timeout", "request_timeout", DEFAULTS["request_timeout"]))
        )
        self.num_ctx_spin = QSpinBox()
        self.num_ctx_spin.setRange(1024, 262144)
        self.num_ctx_spin.setSingleStep(1024)
        self.num_ctx_spin.setSuffix(" tokens")
        self.num_ctx_spin.setToolTip(
            "Maximum context sent to local Ollama models. Lower values use less VRAM."
        )
        self.num_ctx_spin.setValue(
            int(setting("num_ctx", "num_ctx", DEFAULTS["num_ctx"]))
        )
        self.pdf_dpi_spin = QSpinBox()
        self.pdf_dpi_spin.setRange(72, 600)
        self.pdf_dpi_spin.setSingleStep(25)
        self.pdf_dpi_spin.setSuffix(" DPI")
        self.pdf_dpi_spin.setToolTip(
            "Resolution used to rasterize each PDF page before sending it to a vision model."
        )
        self.pdf_dpi_spin.setValue(
            int(setting("pdf_dpi", "pdf_dpi", DEFAULTS["pdf_dpi"]))
        )
        initial_judge = resolve_setting(
            self._settings, f"judge/{provider}", getattr(args, "judge_model", None),
            DEFAULT_JUDGES[provider],
        )
        self.judge_combo = QComboBox()
        self.judge_combo.setEditable(True)
        self.judge_combo.addItem(initial_judge)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_LABELS))
        self.provider_combo.setCurrentText(provider)
        # One API key per provider; the key field edits the current provider's key.
        self._keys = {
            name: setting(f"{name}_key", f"{name}_api_key", "")
            for name in PROVIDER_LABELS
        }
        self.key_edit = QLineEdit(self._keys.get(provider, ""))
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_label = QLabel()
        self.include_paid_checkbox = QCheckBox()
        self.include_paid_checkbox.setChecked(
            resolve_setting(self._settings, "include_paid", None, DEFAULTS["include_paid"])
        )
        self.include_paid_checkbox.toggled.connect(self._on_include_paid_changed)
        self.model_list = QListWidget()
        self.refresh_btn = QPushButton()
        self._apply_provider_labels(provider)
        self.all_btn = QPushButton("Select all")
        self.none_btn = QPushButton("Select none")
        self.run_btn = QPushButton("Run evaluation")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.status = QLabel("Idle")
        self.log = QPlainTextEdit(readOnly=True)
        self.vision_debug_log = QPlainTextEdit(readOnly=True)
        self.judge_debug_log = QPlainTextEdit(readOnly=True)
        for log_widget in (self.log, self.vision_debug_log, self.judge_debug_log):
            log_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            log_widget.customContextMenuRequested.connect(
                lambda position, widget=log_widget: self._show_log_context_menu(
                    widget, position
                )
            )
        self._debug_log_paths = {"vision": None, "judge": None}
        self._debug_log_state = {
            "vision": {"path": None, "position": 0, "size": 0},
            "judge": {"path": None, "position": 0, "size": 0},
        }
        self._debug_log_timer = QTimer(self)
        self._debug_log_timer.setInterval(500)
        self._debug_log_timer.timeout.connect(self._refresh_debug_logs)
        self._debug_log_timer.start()
        self.table = QTableWidget(0, len(RESULT_COLUMNS))
        self.table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setStretchLastSection(True)

        self._build_layout()
        self.refresh_btn.clicked.connect(self.refresh_models)
        self.all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.none_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.run_btn.clicked.connect(self.run_evaluation)
        self.cancel_btn.clicked.connect(self.cancel)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.refresh_models()

    def _api_key(self) -> str:
        return self.key_edit.text().strip()

    def _apply_provider_labels(self, provider: str) -> None:
        label = PROVIDER_LABELS[provider]
        env_var = f"{provider.upper()}_API_KEY"
        self.key_label.setText(f"{label} key")
        self.key_edit.setPlaceholderText(f"{label} API key (or {env_var} env var)")
        self.key_edit.setEnabled(True)
        if provider == "ollama":
            self.url_edit.setToolTip(
                "Use https://ollama.com for direct Ollama Cloud API access, or keep "
                "http://localhost:11434 for your local Ollama server."
            )
        else:
            self.url_edit.setToolTip("")
        self.include_paid_checkbox.setText(f"Show paid {label} vision models")
        self.include_paid_checkbox.setVisible(provider in PAID_CAPABLE)
        source = {
            "ollama": "Ollama", "openrouter": "openrouter.ai",
            "tokenharbor": "tokenharbor.ai", "nvidia": "build.nvidia.com",
                        "gemini": "ai.google.dev",
                    }
        self.refresh_btn.setText(f"Refresh from {source[provider]}")

    def _on_provider_changed(self, provider: str) -> None:
        # Each provider remembers its own judge model and API key.
        self._settings.setValue(f"judge/{self._provider}", self.judge_combo.currentText().strip())
        self._keys[self._provider] = self.key_edit.text().strip()
        self._provider = provider
        saved = self._settings.value(f"judge/{provider}", DEFAULT_JUDGES[provider])
        self.judge_combo.clear()
        self.judge_combo.addItem(saved or DEFAULT_JUDGES[provider])
        self.key_edit.setText(self._keys.get(provider, ""))
        self._apply_provider_labels(provider)
        self.refresh_models()

    def _on_include_paid_changed(self, _checked: bool) -> None:
        if self.provider_combo.currentText() in PAID_CAPABLE:
            self.refresh_models()

    def save_settings(self) -> None:
        s = self._settings
        s.setValue("pdf", self.pdf_edit.text().strip())
        s.setValue("truth", self.truth_edit.text().strip())
        s.setValue("output_dir", self.out_edit.text().strip())
        s.setValue("ollama_url", self.url_edit.text().strip())
        s.setValue("request_timeout", self.request_timeout_spin.value())
        s.setValue("num_ctx", self.num_ctx_spin.value())
        s.setValue("pdf_dpi", self.pdf_dpi_spin.value())
        provider = self.provider_combo.currentText()
        s.setValue("provider", provider)
        self._keys[provider] = self.key_edit.text().strip()
        for name, key in self._keys.items():
            s.setValue(f"{name}_key", key)
        s.setValue("include_paid", self.include_paid_checkbox.isChecked())
        s.setValue(f"judge/{provider}", self.judge_combo.currentText().strip())
        s.sync()

    def closeEvent(self, event) -> None:
        self.save_settings()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        settings = QGroupBox("Inputs")
        grid = QGridLayout(settings)
        for row, (label, edit, browse) in enumerate([
            ("PDF", self.pdf_edit, "pdf"),
            ("Ground truth", self.truth_edit, "txt"),
            ("Output dir", self.out_edit, "dir"),
            ("Provider", self.provider_combo, None),
            (self.key_label, self.key_edit, None),
            ("Ollama URL", self.url_edit, None),
            ("Judge model", self.judge_combo, None),
            ("Request timeout", self.request_timeout_spin, None),
            ("Ollama context cap", self.num_ctx_spin, None),
            ("PDF render resolution", self.pdf_dpi_spin, None),
        ]):
            grid.addWidget(label if isinstance(label, QLabel) else QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            if browse:
                button = QPushButton("Browse...")
                button.clicked.connect(lambda _=False, e=edit, k=browse: self._browse(e, k))
                grid.addWidget(button, row, 2)

        models_box = QGroupBox("Vision models (check to evaluate)")
        models_layout = QVBoxLayout(models_box)
        buttons = QHBoxLayout()
        for button in (self.refresh_btn, self.all_btn, self.none_btn):
            buttons.addWidget(button)
        buttons.addWidget(self.include_paid_checkbox)
        buttons.addStretch()
        models_layout.addLayout(buttons)
        models_layout.addWidget(self.model_list)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.addWidget(self.table)

        log_box = QGroupBox("Progress")
        log_layout = QVBoxLayout(log_box)
        controls = QHBoxLayout()
        controls.addWidget(self.run_btn)
        controls.addWidget(self.cancel_btn)
        controls.addWidget(self.progress, 1)
        log_layout.addLayout(controls)
        log_layout.addWidget(self.status)
        log_tabs = QTabWidget()
        log_tabs.addTab(self.log, "Progress")
        log_tabs.addTab(self.vision_debug_log, "Vision LLM debug")
        log_tabs.addTab(self.judge_debug_log, "Judge LLM debug")
        log_layout.addWidget(log_tabs)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(models_box)
        top.addWidget(log_box)
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(top)
        outer.addWidget(results_box)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(settings)
        layout.addWidget(outer, 1)
        self.setCentralWidget(central)

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select directory")
        else:
            filt = "PDF (*.pdf)" if kind == "pdf" else "Text (*.txt)"
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            edit.setText(path)

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _show_log_context_menu(self, widget: QPlainTextEdit, position) -> None:
        menu = widget.createStandardContextMenu()
        output = widget.toPlainText()
        if output:
            menu.addSeparator()
            copy_all_action = menu.addAction("Copy all output")
            copy_all_action.triggered.connect(
                lambda _checked=False, text=output: (
                    QApplication.clipboard().setText(text)
                )
            )
        menu.exec(widget.mapToGlobal(position))

    def _set_debug_log_paths(self, vision_path: str, judge_path: str) -> None:
        self._debug_log_paths = {"vision": vision_path, "judge": judge_path}
        for kind, widget in (("vision", self.vision_debug_log), ("judge", self.judge_debug_log)):
            self._debug_log_state[kind] = {"path": None, "position": 0, "size": 0}
            widget.clear()
        self._refresh_debug_logs()

    def _refresh_debug_logs(self) -> None:
        for kind, widget in (("vision", self.vision_debug_log), ("judge", self.judge_debug_log)):
            path = self._debug_log_paths[kind]
            if not path:
                continue
            try:
                size = os.path.getsize(path)
                state = self._debug_log_state[kind]
                if state["path"] != path or size < state["size"]:
                    widget.clear()
                    state.update(path=path, position=0, size=0)
                if size == state["size"]:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as log_file:
                    log_file.seek(state["position"])
                    added_text = log_file.read()
                    state["position"] = log_file.tell()
                state["size"] = size
                if added_text:
                    cursor = widget.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertText(added_text)
                    widget.setTextCursor(cursor)
                    widget.ensureCursorVisible()
            except FileNotFoundError:
                continue

    def _set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.num_ctx_spin.setEnabled(not busy)
        self.pdf_dpi_spin.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.model_list.count()):
            self.model_list.item(i).setCheckState(state)

    def _selected_models(self) -> List[str]:
        return [
            self.model_list.item(i).data(Qt.ItemDataRole.UserRole)
            or self.model_list.item(i).text()
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _launch(self, task, on_done, on_fail, on_progress=None) -> None:
        thread, worker = start_worker(self, task)
        self._thread, self._worker = thread, worker
        # Keep running threads referenced so a newer launch cannot destroy them.
        self._running.add(thread)
        thread.finished.connect(lambda t=thread: self._running.discard(t))
        worker.finished.connect(on_done)
        worker.failed.connect(on_fail)
        worker.debug_logs_ready.connect(self._set_debug_log_paths)
        if on_progress:
            worker.progress.connect(on_progress)
        thread.start()

    def refresh_models(self) -> None:
        provider = self.provider_combo.currentText()
        self._fetch_id += 1
        fetch_id = self._fetch_id
        self.model_list.clear()
        self._set_busy(True)
        self.cancel_btn.setEnabled(False)
        include_paid = (
            provider in PAID_CAPABLE and self.include_paid_checkbox.isChecked()
        )
        if provider == "ollama":
            listing = "Ollama vision and judge models"
        else:
            listing = (
                "vision and text judge models"
                if include_paid
                else f"free {provider} vision and text judge models"
            )
        self.status.setText(f"Fetching {listing}...")
        self._launch(
            make_fetch_task(
                self.url_edit.text().strip(),
                provider,
                self._api_key(),
                include_paid=include_paid,
            ),
            lambda result: fetch_id == self._fetch_id and self._on_models(result, provider),
            lambda msg: fetch_id == self._fetch_id and self._on_models_failed(msg),
        )

    def _on_models(self, result, provider: str) -> None:
        models, judge_models, judge_error, vision_error = result
        self._populate_judge_models(judge_models, provider)
        if judge_error:
            self._log(f"Could not refresh {provider} text judge models: {judge_error}")
        if vision_error:
            self._log(vision_error)
        self.model_list.clear()
        free_count = 0
        paid_count = 0
        for model in models:
            if hasattr(model, "display_name"):
                name = model.display_name
                model_id = model.model_id
                is_free = model.is_free
                free_count += int(is_free)
                paid_count += int(not is_free)
            else:
                name = model
                model_id = model
                is_free = True
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, is_free)
            item.setCheckState(
                Qt.CheckState.Checked if is_free else Qt.CheckState.Unchecked
            )
            self.model_list.addItem(item)
        if not models:
            self.status.setText(f"No {provider} vision models are currently available")
            self._log(f"No {provider} vision models are currently available.")
            self.run_btn.setEnabled(False)
        else:
            if provider in PAID_CAPABLE:
                label = PROVIDER_LABELS[provider]
                self.status.setText(
                    f"Found {free_count} free and {paid_count} paid {label} vision models"
                )
                self._log(
                    f"Discovered {free_count} free and {paid_count} paid {label} vision models"
                )
            else:
                qualifier = "" if provider == "ollama" else "free "
                self.status.setText(
                    f"Found {len(models)} {qualifier}{provider} vision models"
                )
                self._log(
                    f"Discovered {len(models)} {qualifier}{provider} vision models"
                )
            self.run_btn.setEnabled(True)
        self._set_busy(False)

    def _populate_judge_models(self, models: List[str], provider: str) -> None:
        selected = self.judge_combo.currentText().strip() or DEFAULT_JUDGES[provider]
        self.judge_combo.clear()
        for model in models:
            self.judge_combo.addItem(model)
        if selected not in models:
            # Keep saved/custom model ids usable if a provider omits them from
            # its current catalog response.
            self.judge_combo.addItem(selected)
        self.judge_combo.setCurrentText(selected)

    def _on_models_failed(self, message: str) -> None:
        self.status.setText("Could not fetch model list")
        self._log(f"Model discovery failed: {message}")
        self._set_busy(False)

    def run_evaluation(self) -> None:
        selected_items = [
            self.model_list.item(i)
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        models = [
            item.data(Qt.ItemDataRole.UserRole) or item.text()
            for item in selected_items
        ]
        pdf, truth = self.pdf_edit.text().strip(), self.truth_edit.text().strip()
        if not models:
            QMessageBox.warning(self, "No models", "Select at least one model.")
            return
        if not pdf or not truth:
            QMessageBox.warning(self, "Missing input", "Choose a PDF and a ground truth file.")
            return
        paid_models = [
            item.data(Qt.ItemDataRole.UserRole) or item.text()
            for item in selected_items
            if item.data(Qt.ItemDataRole.UserRole + 1) is False
        ]
        if paid_models and QMessageBox.question(
            self,
            f"Paid {PROVIDER_LABELS[self.provider_combo.currentText()]} models selected",
            "The selected paid models may incur charges:\n\n"
            + "\n".join(paid_models)
            + "\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self._set_busy(True)
        provider = self.provider_combo.currentText()
        self._log(
            f"Starting evaluation of {len(models)} models; settings: "
            f"provider={provider}, models={', '.join(models)}, "
            f"Ollama context cap={self.num_ctx_spin.value()} tokens, "
            f"PDF render resolution={self.pdf_dpi_spin.value()} DPI, "
            f"request timeout={self.request_timeout_spin.value()} seconds"
        )
        self._set_debug_log_paths("", "")
        task = make_evaluation_task(
            pdf, truth, self.out_edit.text().strip() or "evaluation_results",
            self.url_edit.text().strip(), self.judge_combo.currentText().strip(), models,
            provider, self._api_key(),
            self.request_timeout_spin.value(),
            self.num_ctx_spin.value(), self.pdf_dpi_spin.value(),
            gemini_rate_limits=self._gemini_rate_limits,
        )
        self._launch(task, self._on_results, self._on_eval_failed, self._on_progress)

    def cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.status.setText("Cancelling after the current model...")

    def _on_progress(self, step: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(step)
        self.status.setText(f"{message} ({step}/{total})")
        self._log(message)

    def _on_eval_failed(self, message: str) -> None:
        self.status.setText("Evaluation failed")
        self._log(f"Error: {message}")
        self._set_busy(False)

    def _on_results(self, summary) -> None:
        by_name = {r.model_name: r for r in summary.results}
        self.table.setRowCount(len(summary.ranking))
        for row, (name, score) in enumerate(summary.ranking):
            r = by_name[name]
            values = [
                str(row + 1), name, f"{score:.3f}", f"{r.character_accuracy:.3f}",
                f"{r.semantic_similarity:.3f}", f"{r.word_error_rate:.3f}",
                f"{r.levenshtein_distance:.3f}", f"{r.execution_time:.1f}",
                "; ".join(filter(None, [r.error_message, getattr(r, "judge_error", None)])),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == len(RESULT_COLUMNS) - 1 and value:
                    item.setToolTip(value)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(len(RESULT_COLUMNS) - 1, 360)
        self.table.resizeRowsToContents()
        self.status.setText(f"Done. Best: {summary.best_model} ({summary.best_score:.3f})")
        self._log("Evaluation complete; results saved to the output directory")
        self._set_busy(False)


def run_gui(args=None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
