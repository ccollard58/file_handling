"""PyQt6 GUI for selecting cloud vision models, running the evaluation,
and viewing progress and results."""

import sys
from typing import Callable, List

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ollama_cloud_models import fetch_free_cloud_vision_models
from openrouter_models import fetch_vision_models as fetch_openrouter_models
from tokenharbor_models import fetch_vision_models as fetch_tokenharbor_models
from nvidia_models import fetch_vision_models as fetch_nvidia_models

RESULT_COLUMNS = [
    "Rank", "Model", "Composite", "Char Acc", "Semantic", "WER",
    "Levenshtein", "Time (s)", "Error",
]

PROVIDER_LABELS = {
    "ollama": "Ollama", "openrouter": "OpenRouter", "tokenharbor": "Token Harbor",
    "nvidia": "NVIDIA",
}
PAID_CAPABLE = ("openrouter", "tokenharbor")
DEFAULT_JUDGES = {
    "ollama": "gpt-oss:120b",
    "openrouter": "openai/gpt-oss-120b:free",
    "tokenharbor": "mimo-v2.6-flash:free",
    "nvidia": "z-ai/glm5",
}
DEFAULTS = {
    "pdf": "", "truth": "", "output_dir": "evaluation_results",
    "ollama_url": "http://localhost:11434", "provider": "ollama",
    "include_paid": True,
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
                         provider: str = "ollama", api_key: str = ""):
    def task(worker: TaskWorker):
        from evaluate_handwriting_vision_models import VisionModelEvaluator

        evaluator = VisionModelEvaluator(
            ollama_base_url=url, judge_model=judge, provider=provider,
            ollama_api_key=api_key or None if provider == "ollama" else None,
            openrouter_api_key=api_key or None if provider == "openrouter" else None,
            tokenharbor_api_key=api_key or None if provider == "tokenharbor" else None,
            nvidia_api_key=api_key or None if provider == "nvidia" else None,
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
        return lambda worker: fetch_openrouter_models(
            api_key=api_key or None, include_paid=include_paid
        )
    if provider == "tokenharbor":
        return lambda worker: fetch_tokenharbor_models(
            api_key=api_key or None, include_paid=include_paid
        )
    if provider == "nvidia":
        return lambda worker: fetch_nvidia_models(api_key=api_key or None)
    return lambda worker: fetch_free_cloud_vision_models(
        ollama_url, api_key=api_key or None
    )


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Cloud Vision Model Evaluation")
        self.resize(1100, 800)
        self._thread = None
        self._worker = None
        self._running = set()
        self._fetch_id = 0

        self._settings = new_settings()

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
        self.judge_edit = QLineEdit(
            resolve_setting(
                self._settings, f"judge/{provider}", getattr(args, "judge_model", None),
                DEFAULT_JUDGES[provider],
            )
        )
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
            "ollama": "ollama.com", "openrouter": "openrouter.ai",
            "tokenharbor": "tokenharbor.ai", "nvidia": "build.nvidia.com",
        }
        self.refresh_btn.setText(f"Refresh from {source[provider]}")

    def _on_provider_changed(self, provider: str) -> None:
        # Each provider remembers its own judge model and API key.
        self._settings.setValue(f"judge/{self._provider}", self.judge_edit.text().strip())
        self._keys[self._provider] = self.key_edit.text().strip()
        self._provider = provider
        saved = self._settings.value(f"judge/{provider}", DEFAULT_JUDGES[provider])
        self.judge_edit.setText(saved or DEFAULT_JUDGES[provider])
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
        provider = self.provider_combo.currentText()
        s.setValue("provider", provider)
        self._keys[provider] = self.key_edit.text().strip()
        for name, key in self._keys.items():
            s.setValue(f"{name}_key", key)
        s.setValue("include_paid", self.include_paid_checkbox.isChecked())
        s.setValue(f"judge/{provider}", self.judge_edit.text().strip())
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
            ("Judge model", self.judge_edit, None),
        ]):
            grid.addWidget(label if isinstance(label, QLabel) else QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            if browse:
                button = QPushButton("Browse...")
                button.clicked.connect(lambda _=False, e=edit, k=browse: self._browse(e, k))
                grid.addWidget(button, row, 2)

        models_box = QGroupBox("Free cloud vision models (check to evaluate)")
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
        log_layout.addWidget(self.log)

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

    def _set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
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
        listing = (
            "vision models" if include_paid else f"free {provider} vision models"
        )
        self.status.setText(f"Fetching {listing}...")
        self._launch(
            make_fetch_task(
                self.url_edit.text().strip(),
                provider,
                self._api_key(),
                include_paid=include_paid,
            ),
            lambda models: fetch_id == self._fetch_id and self._on_models(models, provider),
            lambda msg: fetch_id == self._fetch_id and self._on_models_failed(msg),
        )

    def _on_models(self, models, provider: str) -> None:
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
                self.status.setText(f"Found {len(models)} free {provider} vision models")
                self._log(f"Discovered {len(models)} free {provider} vision models")
            self.run_btn.setEnabled(True)
        self._set_busy(False)

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
        self._log(f"Starting evaluation of {len(models)} models")
        task = make_evaluation_task(
            pdf, truth, self.out_edit.text().strip() or "evaluation_results",
            self.url_edit.text().strip(), self.judge_edit.text().strip(), models,
            self.provider_combo.currentText(), self._api_key(),
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
                r.error_message or "",
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
