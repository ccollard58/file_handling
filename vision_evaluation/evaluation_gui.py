"""PyQt6 GUI for selecting Ollama cloud vision models, running the
handwriting evaluation, and viewing progress and results."""

import sys
from typing import Callable, List

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ollama_cloud_models import fetch_free_cloud_vision_models
from openrouter_models import fetch_free_vision_models

RESULT_COLUMNS = [
    "Rank", "Model", "Composite", "Char Acc", "Semantic", "WER",
    "Levenshtein", "Time (s)", "Error",
]


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
            openrouter_api_key=api_key or None,
        )
        return evaluator.evaluate_all_models(
            pdf, truth, out_dir, models=models,
            progress_callback=worker.progress.emit,
            should_cancel=worker.is_cancelled,
        )
    return task


def make_fetch_task(ollama_url: str, provider: str = "ollama", api_key: str = ""):
    if provider == "openrouter":
        return lambda worker: fetch_free_vision_models(api_key=api_key or None)
    return lambda worker: fetch_free_cloud_vision_models(ollama_url)


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Ollama Cloud Vision Model Evaluation")
        self.resize(1100, 800)
        self._thread = None
        self._worker = None

        self.pdf_edit = QLineEdit(getattr(args, "pdf_path", None) or "")
        self.truth_edit = QLineEdit(getattr(args, "ground_truth_path", None) or "")
        self.out_edit = QLineEdit(getattr(args, "output_dir", None) or "evaluation_results")
        self.url_edit = QLineEdit(getattr(args, "ollama_url", None) or "http://localhost:11434")
        self.judge_edit = QLineEdit(getattr(args, "judge_model", None) or "gpt-oss:120b")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "openrouter"])
        self.provider_combo.setCurrentText(getattr(args, "provider", None) or "ollama")
        self.key_edit = QLineEdit(getattr(args, "openrouter_api_key", None) or "")
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("OpenRouter API key (or OPENROUTER_API_KEY env var)")
        self.model_list = QListWidget()
        self.refresh_btn = QPushButton("Refresh from ollama.com")
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

    def _on_provider_changed(self, provider: str) -> None:
        self.judge_edit.setText(
            "openai/gpt-oss-120b:free" if provider == "openrouter" else "gpt-oss:120b"
        )
        self.refresh_models()

    def _build_layout(self) -> None:
        settings = QGroupBox("Inputs")
        grid = QGridLayout(settings)
        for row, (label, edit, browse) in enumerate([
            ("PDF", self.pdf_edit, "pdf"),
            ("Ground truth", self.truth_edit, "txt"),
            ("Output dir", self.out_edit, "dir"),
            ("Provider", self.provider_combo, None),
            ("OpenRouter key", self.key_edit, None),
            ("Ollama URL", self.url_edit, None),
            ("Judge model", self.judge_edit, None),
        ]):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            if browse:
                button = QPushButton("Browse...")
                button.clicked.connect(lambda _=False, e=edit, k=browse: self._browse(e, k))
                grid.addWidget(button, row, 2)

        models_box = QGroupBox("Free Ollama cloud vision models (check to evaluate)")
        models_layout = QVBoxLayout(models_box)
        buttons = QHBoxLayout()
        for button in (self.refresh_btn, self.all_btn, self.none_btn):
            buttons.addWidget(button)
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
            self.model_list.item(i).text()
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _launch(self, task, on_done, on_fail, on_progress=None) -> None:
        self._thread, self._worker = start_worker(self, task)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(on_fail)
        if on_progress:
            self._worker.progress.connect(on_progress)
        self._thread.start()

    def refresh_models(self) -> None:
        self._set_busy(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Fetching free vision models...")
        self._launch(
            make_fetch_task(self.url_edit.text().strip(), self.provider_combo.currentText(), self._api_key()),
            self._on_models,
            self._on_models_failed,
        )

    def _on_models(self, models: List[str]) -> None:
        self.model_list.clear()
        for name in models:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.model_list.addItem(item)
        if not models:
            self.status.setText("No free Ollama cloud vision models are currently advertised")
            self._log("No free Ollama cloud vision models are currently advertised.")
            self.run_btn.setEnabled(False)
        else:
            self.status.setText(f"Found {len(models)} free cloud vision models")
            self._log(f"Discovered {len(models)} free cloud vision models")
            self.run_btn.setEnabled(True)
        self._set_busy(False)

    def _on_models_failed(self, message: str) -> None:
        self.status.setText("Could not fetch model list")
        self._log(f"Model discovery failed: {message}")
        self._set_busy(False)

    def run_evaluation(self) -> None:
        models = self._selected_models()
        pdf, truth = self.pdf_edit.text().strip(), self.truth_edit.text().strip()
        if not models:
            QMessageBox.warning(self, "No models", "Select at least one model.")
            return
        if not pdf or not truth:
            QMessageBox.warning(self, "Missing input", "Choose a PDF and a ground truth file.")
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
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
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
