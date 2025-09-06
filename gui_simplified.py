import os
import sys
import json
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QTreeView, QFileDialog, QCheckBox, QComboBox,
                           QMessageBox, QHeaderView, QSplitter, QProgressDialog,
                           QMenu, QInputDialog, QGridLayout, QGroupBox, QTextEdit, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QStyleOptionButton, QToolButton, QTabWidget, QProgressBar, QAbstractItemView, QSizePolicy)
from PyQt6.QtCore import Qt, QFileInfo, QDir, QModelIndex, QThread, pyqtSignal, QSize, QTimer, QEvent, QRect, QObject, QDateTime, QSortFilterProxyModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFileSystemModel, QAction, QColor, QPixmap, QImage, QIcon, QPainter, QFont, QFontMetrics
from PyQt6.QtPdf import QPdfDocument

class QtLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits Qt signals for real-time log display"""
    
    log_message = pyqtSignal(str, str, str)  # level, time, message
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        
    def emit(self, record):
        """Emit the log record as a Qt signal"""
        try:
            # Format the message
            level = record.levelname
            time_str = self.format_time(record)
            message = record.getMessage()
            
            # Emit the signal
            self.log_message.emit(level, time_str, message)
        except Exception:
            # Ignore errors in logging to prevent infinite loops
            pass
    
    def format_time(self, record):
        """Format the timestamp for display"""
        import time
        return time.strftime('%H:%M:%S', time.localtime(record.created))

class CheckBoxDelegate(QStyledItemDelegate):
    """Custom delegate to center checkboxes in the first column"""
    
    def paint(self, painter, option, index):
        if index.column() == 0:
            # Get the checkbox state from the model
            model = index.model()
            if model is None:
                super().paint(painter, option, index)
                return
                
            checkState = model.data(index, Qt.ItemDataRole.CheckStateRole)
            
            # Calculate centered position for a smaller checkbox
            checkbox_size = 13  # Slightly smaller for better fit
            x = option.rect.x() + (option.rect.width() - checkbox_size) // 2
            y = option.rect.y() + (option.rect.height() - checkbox_size) // 2
            
            # Create centered checkbox rectangle
            checkbox_rect = QRect(x, y, checkbox_size, checkbox_size)
            
            # Draw background if selected
            if option.state & QStyle.StateFlag.State_Selected:
                if painter is not None:
                    painter.fillRect(option.rect, option.palette.highlight())
            
            # Create style option for checkbox
            checkbox_style = QStyleOptionButton()
            checkbox_style.rect = checkbox_rect
            checkbox_style.state = QStyle.StateFlag.State_Enabled
            
            if checkState == Qt.CheckState.Checked:
                checkbox_style.state |= QStyle.StateFlag.State_On
            else:
                checkbox_style.state |= QStyle.StateFlag.State_Off
              # Draw the checkbox using the application style
            style = QApplication.style()
            if style is not None:
                style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorCheckBox, checkbox_style, painter)
        else:
            super().paint(painter, option, index)
    
    def editorEvent(self, event, model, option, index):
        """Handle mouse clicks on the checkbox"""
        if index.column() == 0 and event is not None and event.type() == QEvent.Type.MouseButtonRelease:
            # Toggle the checkbox state
            if model is None:
                return super().editorEvent(event, model, option, index)
                
            current_state = model.data(index, Qt.ItemDataRole.CheckStateRole)
            new_state = Qt.CheckState.Unchecked if current_state == Qt.CheckState.Checked else Qt.CheckState.Checked
            return model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
        return super().editorEvent(event, model, option, index)

class ResultSortProxyModel(QSortFilterProxyModel):
    """Proxy model to enable sorting by column with proper type handling (e.g., dates)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        # Column mapping based on headers:
        # 0: checkbox, 1: Original Filename, 2: New Filename, 3: Destination Folder,
        # 4: Identity, 5: Date, 6: Description
        column = left.column()

        # For checkbox column sort unchecked before checked
        if column == 0:
            ls = self.sourceModel().data(left, Qt.ItemDataRole.CheckStateRole)
            rs = self.sourceModel().data(right, Qt.ItemDataRole.CheckStateRole)
            # Treat None as unchecked
            lv = 1 if ls == Qt.CheckState.Checked else 0
            rv = 1 if rs == Qt.CheckState.Checked else 0
            return lv < rv

        # Fetch display data
        ldata = self.sourceModel().data(left, Qt.ItemDataRole.DisplayRole) or ""
        rdata = self.sourceModel().data(right, Qt.ItemDataRole.DisplayRole) or ""

        # Date column: attempt to parse into comparable tuples
        if column == 5:
            from datetime import datetime
            def parse_date(s: str):
                s = s.strip()
                if not s:
                    return None
                # Try a few common patterns
                patterns = [
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%m/%d/%Y",
                    "%d/%m/%Y",
                    "%b %d, %Y",
                    "%B %d, %Y",
                    "%Y-%m-%d %H:%M:%S",
                ]
                # fromisoformat
                try:
                    return datetime.fromisoformat(s)
                except Exception:
                    pass
                for p in patterns:
                    try:
                        return datetime.strptime(s, p)
                    except Exception:
                        continue
                return None

            ld = parse_date(ldata)
            rd = parse_date(rdata)
            if ld is not None and rd is not None:
                return ld < rd
            # Fallback to string compare

        # Numeric-aware fallback: if both are integers, compare numerically
        if isinstance(ldata, str) and ldata.isdigit() and isinstance(rdata, str) and rdata.isdigit():
            try:
                return int(ldata) < int(rdata)
            except Exception:
                pass

        # Default: case-insensitive string compare
        return str(ldata).lower() < str(rdata).lower()

class FileFinderThread(QThread):
    """Thread to find files in background without freezing the UI"""
    file_found = pyqtSignal(str)
    finished = pyqtSignal(int)  # Signal to indicate completion with count of files found
    
    def __init__(self, root_path, extensions):
        super().__init__()
        self.root_path = root_path
        self.extensions = extensions
        self.abort = False
    
    def run(self):
        count = 0
        for root, _, files in os.walk(self.root_path):
            if self.abort:
                break
                
            for file in files:
                if self.abort:
                    break
                    
                if any(file.lower().endswith(ext) for ext in self.extensions):
                    file_path = os.path.join(root, file)
                    self.file_found.emit(file_path)
                    count += 1
        
        self.finished.emit(count)
    
    def stop(self):
        self.abort = True

class FileAnalysisThread(QThread):
    """Thread to analyze files in background without freezing the UI"""
    file_analyzed = pyqtSignal(dict)  # Emits analysis data for each file
    progress_updated = pyqtSignal(int, str)  # Progress value and current file
    analysis_finished = pyqtSignal()
    analysis_error = pyqtSignal(str, str)  # file_path, error_message
    
    def __init__(self, file_list, document_processor, llm_analyzer, file_handler):
        super().__init__()
        self.file_list = file_list
        self.document_processor = document_processor
        self.llm_analyzer = llm_analyzer
        self.file_handler = file_handler
        self.abort = False
    
    def run(self):
        for i, file_path in enumerate(self.file_list):
            if self.abort:
                break
            
            # Update progress
            self.progress_updated.emit(i, os.path.basename(file_path))
            
            try:
                # Extract text from document
                extracted_text = self.document_processor.extract_text(file_path)
                
                # Get file creation date
                creation_date = self.file_handler.get_file_creation_date(file_path)
                
                # Analyze document with LLM
                analysis_result = self.llm_analyzer.analyze_document(
                    extracted_text, 
                    os.path.basename(file_path),
                    creation_date,
                    file_path  # Pass the full file path for image analysis
                )
                
                # Generate new filename
                new_filename = self.file_handler.generate_new_filename(
                    analysis_result, 
                    os.path.basename(file_path)
                )
                
                # Get destination path
                destination_path = self.file_handler.get_destination_path(analysis_result)
                
                # Store analysis result
                analysis_data = {
                    'original_path': file_path,
                    'new_filename': new_filename,
                    'destination_folder': os.path.relpath(destination_path, self.file_handler.base_output_dir),
                    'identity': analysis_result['identity'],
                    'date': analysis_result['date'],
                    'description': analysis_result['description'],
                    'category': analysis_result['category'],
                    'extracted_text': extracted_text  # Store the extracted text for preview
                }
                
                # Emit the analysis result
                self.file_analyzed.emit(analysis_data)
                
            except Exception as e:
                self.analysis_error.emit(file_path, str(e))
                continue
        
        # Signal completion
        self.analysis_finished.emit()
    
    def stop(self):
        self.abort = True

class FileOrganizerGUI(QMainWindow):
    def __init__(self, document_processor, llm_analyzer, file_handler):
        super().__init__()
        self.document_processor = document_processor
        self.llm_analyzer = llm_analyzer
        self.file_handler = file_handler
        self.analyzed_files = []  # Will store analysis results
        self.current_folder = None  # Current selected folder
        self.analysis_thread = None  # Background analysis thread
        
        # Configuration file path
        self.config_file = os.path.join(os.path.expanduser("~"), ".document_organizer_config.json")
        
        # Load saved configuration
        self.config = self.load_config()
        
        # Set default output folder from config or fallback
        default_output = self.config.get("output_folder", r"E:\scanned documents")
        self.file_handler.base_output_dir = default_output
        self.file_handler._ensure_directories()
        
        # Set default source folder from config
        default_source = self.config.get("source_folder", r"E:\Documents")
        
        # Set up logging handler for real-time display
        self.setup_logging()
        
        self.init_ui()
        self.output_folder_edit.setText(self.file_handler.base_output_dir)  # Show configured folder
        
        # Set the default source folder
        self.set_default_source_folder(default_source)
        
        # Restore LLM settings from config
        self.restore_llm_settings()
        
        # Refresh the GUI controls with the restored settings
        self.populate_settings()
    
    def setup_logging(self):
        """Set up the Qt logging handler for real-time log display"""
        self.qt_log_handler = QtLogHandler()
        self.qt_log_handler.log_message.connect(self.add_log_message)
        
        # Add the handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.qt_log_handler)
        
        # Set the level for our handler from config or default to INFO
        log_level_str = self.config.get("log_level", "INFO")
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.qt_log_handler.setLevel(level_map.get(log_level_str, logging.INFO))
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    logging.info(f"Loaded configuration from {self.config_file}")
                    return config
            else:
                logging.info("No configuration file found, using defaults")
                return {}
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return {}
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            # Get current log level from logging tab if available
            current_log_level = "INFO"  # default
            if hasattr(self, 'log_level_combo'):
                current_log_level = self.log_level_combo.currentText()
            elif "log_level" in self.config:
                current_log_level = self.config["log_level"]
            
            config = {
                "output_folder": self.file_handler.base_output_dir,
                "source_folder": self.current_folder,
                "log_level": current_log_level,
                "llm_settings": {
                    "model": self.llm_analyzer.model,
                    "vision_model": self.llm_analyzer.vision_model,
                    "temperature": self.llm_analyzer.temperature
                }
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                logging.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def restore_llm_settings(self):
        """Restore LLM settings from configuration."""
        try:
            llm_settings = self.config.get("llm_settings", {})
            if llm_settings:
                model = llm_settings.get("model", self.llm_analyzer.model)
                vision_model = llm_settings.get("vision_model", self.llm_analyzer.vision_model)
                temperature = llm_settings.get("temperature", self.llm_analyzer.temperature)
                
                # Update LLM analyzer with saved settings
                self.llm_analyzer.update_settings(model, temperature, vision_model)
                logging.info(f"Restored LLM settings - Model: {model}, Vision: {vision_model}, Temp: {temperature}")
        except Exception as e:
            logging.error(f"Error restoring LLM settings: {e}")
    
    def init_ui(self):
        """Initialize the tabbed user interface"""
        self.setWindowTitle("Document Organizer - Simplified")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_main_tab()
        self.create_settings_tab()
        self.create_logging_tab()
        
    def create_main_tab(self):
        """Create the main document processing tab"""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "Document Processing")
        
        main_layout = QVBoxLayout(main_tab)
        
        # Step 1: Source Folder Selection
        source_widget = QWidget()
        source_widget.setMaximumHeight(35)
        source_layout = QHBoxLayout(source_widget)
        source_layout.setContentsMargins(5, 5, 5, 5)
        
        source_layout.addWidget(QLabel("Source Folder:"))
        self.source_folder_edit = QLineEdit()
        source_layout.addWidget(self.source_folder_edit)
        
        self.browse_source_btn = QPushButton("Browse...")
        self.browse_source_btn.clicked.connect(self.browse_source_folder)
        source_layout.addWidget(self.browse_source_btn)
        
        main_layout.addWidget(source_widget)
        
        # Step 2: Output Folder Selection (Compact)
        output_widget = QWidget()
        output_widget.setMaximumHeight(35)  # Limit height
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_edit = QLineEdit()
        output_layout.addWidget(self.output_folder_edit)
        
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.browse_output_btn)
        
        main_layout.addWidget(output_widget)
        
        # Create splitter for two-panel layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Step 2: Folder Structure Panel (Simplified)
        folder_panel = QWidget()
        folder_layout = QVBoxLayout(folder_panel)
        
        # File tree view
        self.file_tree = QTreeView()
        self.file_system_model = QFileSystemModel()
        self.file_tree.setModel(self.file_system_model)
        self.file_tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        folder_layout.addWidget(self.file_tree)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.analyze_selected_btn = QPushButton("Analyze Selected Files")
        self.analyze_selected_btn.setEnabled(False)
        self.analyze_selected_btn.clicked.connect(self.analyze_selected_files)
        buttons_layout.addWidget(self.analyze_selected_btn)
        
        self.analyze_all_btn = QPushButton("Analyze All Files in Folder")
        self.analyze_all_btn.setEnabled(False)
        self.analyze_all_btn.clicked.connect(self.analyze_all_files_in_folder)
        buttons_layout.addWidget(self.analyze_all_btn)
        
        folder_layout.addLayout(buttons_layout)
        
        splitter.addWidget(folder_panel)
        
        # Step 3: Results Panel
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        results_layout.addWidget(QLabel("Analysis Results"))
        
        # File list view for results
        self.file_view = QTreeView()
        self.file_model = QStandardItemModel()
        self.file_model.setHorizontalHeaderLabels([
            "✓", "Original Filename", "New Filename", "Destination Folder", 
            "Identity", "Date", "Description"
        ])
        # Sortable proxy model
        self.proxy_model = ResultSortProxyModel(self)
        self.proxy_model.setSourceModel(self.file_model)
        self.file_view.setModel(self.proxy_model)
        self.file_view.setSortingEnabled(True)
        # Ensure horizontal scrollbar remains available even during column resizes
        self.file_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.file_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        
        # Set custom delegate for centering checkboxes
        checkbox_delegate = CheckBoxDelegate()
        self.file_view.setItemDelegateForColumn(0, checkbox_delegate)
        
        # Enable context menu
        self.file_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_view.customContextMenuRequested.connect(self.show_context_menu)
        
        results_layout.addWidget(self.file_view)
        
        # Add progress bar for non-modal analysis progress
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        # Keep progress bar text static to avoid width jitter
        self.progress_bar.setFormat("%v/%m")
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        progress_layout.addWidget(self.progress_bar, 1)

        # Separate label for current filename to avoid resizing the bar
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_label.setToolTip("Current file being analyzed")
        # Slightly deemphasize
        self.progress_label.setStyleSheet("QLabel { color: #666; }")
        progress_layout.addWidget(self.progress_label, 2)
        
        self.cancel_analysis_btn = QPushButton("Cancel")
        self.cancel_analysis_btn.setVisible(False)
        self.cancel_analysis_btn.clicked.connect(self.cancel_analysis)
        progress_layout.addWidget(self.cancel_analysis_btn)
        
        progress_widget.setVisible(False)
        self.progress_widget = progress_widget  # Store reference to show/hide entire progress area
        results_layout.addWidget(progress_widget)
        
        # Process controls
        process_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_files)
        process_layout.addWidget(self.select_all_btn)
        
        self.unselect_all_btn = QPushButton("Unselect All")
        self.unselect_all_btn.clicked.connect(self.unselect_all_files)
        process_layout.addWidget(self.unselect_all_btn)
        
        process_layout.addStretch()
        
        self.process_btn = QPushButton("Process Selected Files")
        self.process_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.process_btn.clicked.connect(self.process_files)
        process_layout.addWidget(self.process_btn)
        
        # Add undo controls
        self.undo_last_btn = QPushButton("Undo Last Action")
        self.undo_last_btn.clicked.connect(self.undo_last_action_gui)
        process_layout.addWidget(self.undo_last_btn)
        
        self.undo_all_btn = QPushButton("Undo All Actions")
        self.undo_all_btn.clicked.connect(self.undo_all_actions_gui)
        process_layout.addWidget(self.undo_all_btn)
        
        results_layout.addLayout(process_layout)
        
        # Add preview panel
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Image preview label
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Reserve only thumbnail dimensions to minimize empty space
        self.preview_image_label.setFixedSize(100, 100)
        self.preview_image_label.setVisible(False)
        preview_layout.addWidget(self.preview_image_label)
        
        # Text preview area
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMinimumHeight(150)
        self.preview_text.setText("Select a file to see preview.")
        preview_layout.addWidget(self.preview_text)
        
        results_layout.addWidget(preview_group)
        
        splitter.addWidget(results_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)
        
        # Configure file view headers
        header = self.file_view.header()
        if header is not None:
            # Fix the checkbox column (0) width and prevent resizing
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(0, 40)  # Made wider to accommodate centered checkbox
            # Configure other columns: interactive resizing
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
            # Do not stretch the last section; allow horizontal scrolling instead
            header.setStretchLastSection(False)
            header.setSortIndicatorShown(True)
        
        # Connect selection change to preview update (only if selectionModel exists)
        selection_model = self.file_view.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self.update_preview)
        else:
            # Defer connection until the event loop starts, to ensure selectionModel is available
            def connect_when_ready():
                selection_model = self.file_view.selectionModel()
                if selection_model is not None:
                    selection_model.selectionChanged.connect(self.update_preview)
            QTimer.singleShot(0, connect_when_ready)
    
    def create_settings_tab(self):
        """Create the settings configuration tab"""
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "Settings")
        
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        settings_layout.setSpacing(20)
        
        # LLM Settings Group
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QGridLayout(llm_group)
        
        # Model selection
        llm_layout.addWidget(QLabel("Text Model:"), 0, 0)
        self.model_combo = QComboBox()
        llm_layout.addWidget(self.model_combo, 0, 1)
        
        # Refresh models button
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.clicked.connect(self.populate_models)
        llm_layout.addWidget(self.refresh_models_btn, 0, 2)
        
        # Vision model selection
        llm_layout.addWidget(QLabel("Vision Model:"), 1, 0)
        self.vision_model_combo = QComboBox()
        llm_layout.addWidget(self.vision_model_combo, 1, 1)
        
        # Refresh vision models button
        self.refresh_vision_models_btn = QPushButton("Refresh")
        self.refresh_vision_models_btn.clicked.connect(self.populate_vision_models)
        llm_layout.addWidget(self.refresh_vision_models_btn, 1, 2)
        
        # Temperature setting
        llm_layout.addWidget(QLabel("Temperature:"), 2, 0)
        self.temp_edit = QLineEdit()
        self.temp_edit.setPlaceholderText("e.g., 0.6")
        llm_layout.addWidget(self.temp_edit, 2, 1)
        
        settings_layout.addWidget(llm_group)
        
        # Apply button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.apply_settings_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }")
        button_layout.addWidget(self.apply_settings_btn)
        button_layout.addStretch()
        
        settings_layout.addLayout(button_layout)
        settings_layout.addStretch()
        
        # Populate initial values
        self.populate_settings()
        
        # Connect change signals to track when settings are modified
        self.model_combo.currentTextChanged.connect(self.on_settings_changed)
        self.vision_model_combo.currentTextChanged.connect(self.on_settings_changed)
        self.temp_edit.textChanged.connect(self.on_settings_changed)
    
    def create_logging_tab(self):
        """Create the real-time logging tab"""
        logging_tab = QWidget()
        self.tab_widget.addTab(logging_tab, "Logging")
        
        logging_layout = QVBoxLayout(logging_tab)
        
        # Add header with controls
        header_layout = QHBoxLayout()
        
        # Log level filter
        header_layout.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        
        # Set initial log level from config
        initial_log_level = self.config.get("log_level", "INFO")
        self.log_level_combo.setCurrentText(initial_log_level)
        self.log_level_combo.currentTextChanged.connect(self.filter_log_messages)
        header_layout.addWidget(self.log_level_combo)
        
        header_layout.addStretch()
        
        # Clear logs button
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        header_layout.addWidget(self.clear_logs_btn)
        
        # Save logs button
        self.save_logs_btn = QPushButton("Save Logs...")
        self.save_logs_btn.clicked.connect(self.save_logs)
        header_layout.addWidget(self.save_logs_btn)
        
        logging_layout.addLayout(header_layout)
        
        # Create the log display area
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(400)
        
        # Set font to monospace for better alignment
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_display.setFont(font)
        
        # Set a dark theme for the log display
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #3e3e3e;
            }
        """)
        
        logging_layout.addWidget(self.log_display)
        
        # Initialize log storage for filtering
        self.log_messages = []
        self.current_log_level = self.config.get("log_level", "INFO")
        
        # Add initial message
        self.add_log_message("INFO", "00:00:00", "Logging tab initialized. Real-time logs will appear here.")
    
    def add_log_message(self, level, time_str, message):
        """Add a log message to the display"""
        # Store the log message
        log_entry = {
            'level': level,
            'time': time_str,
            'message': message
        }
        self.log_messages.append(log_entry)
        
        # Keep only the last 1000 messages to prevent memory issues
        if len(self.log_messages) > 1000:
            self.log_messages = self.log_messages[-1000:]
        
        # Check if this message should be displayed based on current filter
        if self.should_display_log(level):
            self.append_log_to_display(level, time_str, message)
    
    def should_display_log(self, level):
        """Check if a log message should be displayed based on current filter"""
        level_hierarchy = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        current_level_num = level_hierarchy.get(self.current_log_level, 1)
        message_level_num = level_hierarchy.get(level, 1)
        
        return message_level_num >= current_level_num
    
    def append_log_to_display(self, level, time_str, message):
        """Append a formatted log message to the display"""
        # Color code by level
        color_map = {
            "DEBUG": "#888888",
            "INFO": "#ffffff",
            "WARNING": "#ffaa00",
            "ERROR": "#ff6666",
            "CRITICAL": "#ff0000"
        }
        
        color = color_map.get(level, "#ffffff")
        
        # Format the log entry
        formatted_message = f'<span style="color: {color};">[{time_str}] {level:8} | {message}</span>'
        
        # Add to display
        self.log_display.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)
    
    def filter_log_messages(self, level):
        """Filter log messages by level"""
        self.current_log_level = level
        
        # Clear display and re-add filtered messages
        self.log_display.clear()
        
        for log_entry in self.log_messages:
            if self.should_display_log(log_entry['level']):
                self.append_log_to_display(
                    log_entry['level'],
                    log_entry['time'],
                    log_entry['message']
                )
    
    def clear_logs(self):
        """Clear all log messages"""
        self.log_messages.clear()
        self.log_display.clear()
        self.add_log_message("INFO", "00:00:00", "Logs cleared.")
    
    def save_logs(self):
        """Save current log messages to a file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Logs",
                f"document_organizer_logs_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Document Organizer - Application Logs\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for log_entry in self.log_messages:
                        f.write(f"[{log_entry['time']}] {log_entry['level']:8} | {log_entry['message']}\n")
                
                QMessageBox.information(self, "Logs Saved", f"Logs saved to: {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save logs: {str(e)}")
    
    def on_settings_changed(self):
        """Called when any setting is changed to indicate changes need to be applied."""
        self.apply_settings_btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; font-weight: bold; }")
        self.apply_settings_btn.setText("Apply Settings*")
    
    def reset_apply_button(self):
        """Reset the apply button to normal appearance after applying settings."""
        self.apply_settings_btn.setStyleSheet("")
        self.apply_settings_btn.setText("Apply Settings")

    def populate_settings(self):
        """Populates the settings controls with current values."""
        self.populate_models()
        self.populate_vision_models()
        self.temp_edit.setText(str(self.llm_analyzer.temperature))
        
        self.reset_apply_button()  # Reset button when populating with current values

    def populate_models(self):
        """Populates the model dropdown."""
        self.model_combo.clear()
        try:
            # Use the static method to get text models (excludes vision models)
            available_models = self.llm_analyzer.get_text_models()
            if available_models:
                # Sort models alphabetically
                available_models.sort()
                self.model_combo.addItems(available_models)
                current_model = self.llm_analyzer.model
                if current_model in available_models:
                    self.model_combo.setCurrentText(current_model)
                else:
                    # If current model not in list, add it to avoid confusion
                    self.model_combo.addItem(current_model)
                    self.model_combo.setCurrentText(current_model)
            else:
                self.model_combo.addItem("No models found")
                QMessageBox.warning(self, "Ollama Models", "Could not fetch models from Ollama. Is it running?")
        except Exception as e:
            self.model_combo.addItem("Error fetching models")
            QMessageBox.critical(self, "Ollama Error", f"An error occurred while fetching models: {e}")

    def populate_vision_models(self):
        """Populates the vision model dropdown."""
        self.vision_model_combo.clear()
        try:
            # Use the static method to get vision models
            available_models = self.llm_analyzer.get_vision_models()
            if available_models:
                # Sort models alphabetically
                available_models.sort()
                self.vision_model_combo.addItems(available_models)
                current_model = self.llm_analyzer.vision_model
                if current_model in available_models:
                    self.vision_model_combo.setCurrentText(current_model)
                else:
                    # If current model not in list, add it to avoid confusion
                    self.vision_model_combo.addItem(current_model)
                    self.vision_model_combo.setCurrentText(current_model)
            else:
                self.vision_model_combo.addItem("No vision models found")
                QMessageBox.warning(self, "Ollama Vision Models", "Could not fetch vision models from Ollama. Consider installing a vision model like 'llava'.")
        except Exception as e:
            self.vision_model_combo.addItem("Error fetching vision models")
            QMessageBox.critical(self, "Ollama Error", f"An error occurred while fetching vision models: {e}")

    def apply_settings(self):
        """Applies the new settings to the LLM analyzer."""
        model = self.model_combo.currentText()
        vision_model = self.vision_model_combo.currentText()
        temperature_str = self.temp_edit.text()
        
        try:
            temperature = float(temperature_str)
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Temperature", f"Please enter a valid number for temperature (e.g., 0.6).\n{e}")
            return
            
        try:
            # Update LLM settings
            self.llm_analyzer.update_settings(model, temperature, vision_model)
            
            self.reset_apply_button()  # Reset button appearance after successful apply
            QMessageBox.information(self, "Settings Updated", 
                f"Settings updated successfully.\n"
                f"Text Model: {model}\n"
                f"Vision Model: {vision_model}\n"
                f"Temperature: {temperature}")
        except Exception as e:
            QMessageBox.critical(self, "Error Updating Settings", f"Failed to update settings: {e}")
            
        # Save configuration after applying settings
        self.save_config()
    
    def apply_log_level_setting(self, log_level):
        """Apply the log level setting to both the GUI and logging system."""
        try:
            # Update the logging tab filter
            if hasattr(self, 'log_level_combo'):
                self.log_level_combo.setCurrentText(log_level)
                self.filter_log_messages(log_level)
            
            # Update the Qt log handler level
            if hasattr(self, 'qt_log_handler'):
                level_map = {
                    "DEBUG": logging.DEBUG,
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL
                }
                self.qt_log_handler.setLevel(level_map.get(log_level, logging.INFO))
            
            # Store the log level in config for persistence
            self.config["log_level"] = log_level
            
            logging.info(f"Log level updated to: {log_level}")
            
        except Exception as e:
            logging.error(f"Error applying log level setting: {e}")

    def browse_source_folder(self):
        """Browse for source folder and populate file tree"""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.current_folder = folder
            self.source_folder_edit.setText(folder)
            
            # Set up file system model
            self.file_system_model.setRootPath(folder)
            self.file_tree.setRootIndex(self.file_system_model.index(folder))
            
            # Enable action buttons
            self.analyze_selected_btn.setEnabled(True)
            self.analyze_all_btn.setEnabled(True)
            
            # Hide unnecessary columns
            self.file_tree.hideColumn(1)  # Size
            self.file_tree.hideColumn(2)  # Type
            self.file_tree.hideColumn(3)  # Date Modified
            
            # Save configuration
            self.save_config()

    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)
            self.file_handler.base_output_dir = folder
            self.file_handler._ensure_directories()
            
            # Save configuration
            self.save_config()
    
    def get_selected_files(self):
        """Get list of selected files or folders and return supported files"""
        selected_files = []
        selection_model = self.file_tree.selectionModel()
        if selection_model is None:
            return selected_files
        selection = selection_model.selectedIndexes()
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.docx', '.doc', '.xlsx']
        for index in selection:
            if index.column() == 0:  # Avoid duplicates
                file_path = self.file_system_model.filePath(index)
                if os.path.isdir(file_path):
                    # If a folder is selected, include all supported files under it
                    for root, _, files in os.walk(file_path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in supported_extensions):
                                selected_files.append(os.path.join(root, file))
                elif os.path.isfile(file_path):
                    # If a file is selected, include it if supported
                    if any(file_path.lower().endswith(ext) for ext in supported_extensions):
                        selected_files.append(file_path)
        return selected_files
    
    def get_all_files_in_folder(self):
        """Get all supported files in the current folder"""
        if not self.current_folder:        return []
        
        all_files = []
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.docx', '.doc', '.xlsx']
        
        for root, _, files in os.walk(self.current_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    all_files.append(os.path.join(root, file))
        
        return all_files
    
    def analyze_selected_files(self):
        """Analyze the selected files"""
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "No Files Selected", "Please select files to analyze.")
            return
        
        self.analyze_files(selected_files)
    
    def analyze_all_files_in_folder(self):
        """Analyze all supported files in the current folder"""
        if not self.current_folder:
            QMessageBox.warning(self, "No Folder Selected", "Please select a source folder first.")
            return
        
        all_files = self.get_all_files_in_folder()
        if not all_files:
            QMessageBox.information(self, "No Files Found", "No supported files found in the selected folder.")
            return
        
        self.analyze_files(all_files)
    
    def analyze_files(self, file_list):
        """Analyze a list of files and populate results using background thread"""
        if not file_list:
            return
        
        # If an analysis is already running, don't start another
        if self.analysis_thread and self.analysis_thread.isRunning():
            QMessageBox.warning(self, "Analysis in Progress", "An analysis is already running. Please wait for it to complete.")
            return
        
        # Clear previous results
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels([
            "✓", "Original Filename", "New Filename", "Destination Folder", 
            "Identity", "Date", "Description"
        ])
        self.analyzed_files = []
        
        # Show progress bar and cancel button
        self.progress_widget.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.cancel_analysis_btn.setVisible(True)
        self.progress_bar.setMaximum(len(file_list))
        self.progress_bar.setValue(0)
        # Keep format fixed; move filename to adjacent label
        self.progress_bar.setFormat("%v/%m")
        self._current_progress_filename = "Starting analysis…"
        self._update_progress_label()
        
        # Disable analysis buttons during processing
        self.analyze_selected_btn.setEnabled(False)
        self.analyze_all_btn.setEnabled(False)
        
        # Create and start analysis thread
        self.analysis_thread = FileAnalysisThread(
            file_list, 
            self.document_processor, 
            self.llm_analyzer, 
            self.file_handler
        )
        
        # Connect thread signals
        self.analysis_thread.file_analyzed.connect(self.on_file_analyzed)
        self.analysis_thread.progress_updated.connect(self.on_analysis_progress)
        self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)
        
        # Start the thread
        self.analysis_thread.start()
    
    def on_file_analyzed(self, analysis_data):
        """Handle a completed file analysis"""
        self.analyzed_files.append(analysis_data)
        self.add_file_to_model(analysis_data)
    
    def on_analysis_progress(self, progress_value, current_file):
        """Update progress bar during analysis"""
        self.progress_bar.setValue(progress_value)
        # Update filename in separate label and keep bar text static
        self._current_progress_filename = current_file
        self._update_progress_label()
    
    def on_analysis_error(self, file_path, error_message):
        """Handle analysis errors"""
        logging.error(f"Error analyzing {file_path}: {error_message}")
    
    def on_analysis_finished(self):
        """Handle completion of analysis"""
        # Hide progress bar and cancel button
        self.progress_widget.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_analysis_btn.setVisible(False)
        self.progress_label.setVisible(False)
        
        # Re-enable analysis buttons
        self.analyze_selected_btn.setEnabled(True)
        self.analyze_all_btn.setEnabled(True)
        
        # Configure view after adding data
        header = self.file_view.header()
        if header is not None:
            # Fix the checkbox column (0) width and prevent resizing
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(0, 40)  # Made wider to accommodate centered checkbox
            # Configure other columns: interactive resizing
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
            # Do not stretch the last section; allow horizontal scrolling instead
            header.setStretchLastSection(False)

        # Update preview if available
        self.update_preview()
        
        # Show completion message
        QMessageBox.information(self, "Analysis Complete", f"Analysis completed. {len(self.analyzed_files)} files processed.")
        
        # Clean up thread
        self._cleanup_analysis_thread()
    
    def _cleanup_analysis_thread(self):
        """Safely clean up the analysis thread"""
        if self.analysis_thread:
            if self.analysis_thread.isRunning():
                self.analysis_thread.stop()
                if not self.analysis_thread.wait(3000):  # Wait up to 3 seconds
                    self.analysis_thread.terminate()
                    self.analysis_thread.wait()  # Wait for termination to complete
            
            self.analysis_thread.deleteLater()
            self.analysis_thread = None
    
    def cancel_analysis(self):
        """Cancel the running analysis"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                "Cancel Analysis", 
                "Are you sure you want to cancel the analysis?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Hide progress bar and cancel button
                self.progress_widget.setVisible(False)
                self.progress_bar.setVisible(False)
                self.cancel_analysis_btn.setVisible(False)
                self.progress_label.setVisible(False)
                
                # Re-enable analysis buttons
                self.analyze_selected_btn.setEnabled(True)
                self.analyze_all_btn.setEnabled(True)
                
                # Clean up thread
                self._cleanup_analysis_thread()
                
                QMessageBox.information(self, "Analysis Cancelled", "Analysis has been cancelled.")
    
    def get_extracted_text_for_file(self, file_path):
        """Get the stored extracted text for a file from analysis data"""
        for analysis_data in self.analyzed_files:
            if analysis_data['original_path'] == file_path:
                return analysis_data.get('extracted_text', None)
        return None
    
    def update_preview(self):
        """Update the preview panel with the selected file content"""
        if not self.analyzed_files:
            self.preview_text.setText("No files analyzed yet.")
            self.preview_image_label.setVisible(False)
            return
            
        selection_model = self.file_view.selectionModel()
        if selection_model is None:
            self.preview_text.setText("Select a file to see preview.")
            self.preview_image_label.setVisible(False)
            return

        indexes = selection_model.selectedRows()
        if not indexes:
            self.preview_text.setText("Select a file to see preview.")
            self.preview_image_label.setVisible(False)
            return
            
        # Map from proxy index (view) to source model row index
        proxy_index = indexes[0]
        source_index = self.proxy_model.mapToSource(proxy_index)
        row_index = source_index.row()
        if 0 <= row_index < len(self.analyzed_files):
            file_path = self.analyzed_files[row_index]['original_path']
            self.show_file_preview(file_path)
    
    def show_file_preview(self, file_path):
        """Show a preview of the selected file"""
        if not os.path.exists(file_path):
            self.preview_text.setText(f"File not found: {file_path}")
            self.preview_image_label.setVisible(False)
            return
        
        # Try to get extracted text from analysis data first
        extracted_text = self.get_extracted_text_for_file(file_path)
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle image files
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']:
            try:
                # Load image and create a thumbnail
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    # Scale to fit preview area while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(
                        self.preview_image_label.width(), 
                        200,  # Max height
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.preview_image_label.setPixmap(scaled_pixmap)
                    self.preview_image_label.setVisible(True)
                    
                    # For images, show basic file info in text area
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    image = QImage(file_path)
                    dimensions = f"{image.width()}x{image.height()}"
                    
                    # Use stored extracted text if available, otherwise extract (fallback)
                    if extracted_text is None:
                        extracted_text = self.document_processor.extract_text(file_path)
                    
                    preview_text = (
                        f"Image Preview: {os.path.basename(file_path)}\n"
                        f"Dimensions: {dimensions}\n"
                        f"Size: {file_size:.1f} KB\n"
                        f"Format: {file_ext[1:].upper()}\n\n"
                        f"Extracted Text (OCR):\n"
                        f"{extracted_text[:5000]}{'...' if len(extracted_text) > 5000 else ''}"
                    )
                    self.preview_text.setText(preview_text)
                else:
                    self.preview_text.setText(f"Cannot load image: {file_path}")
                    self.preview_image_label.setVisible(False)
            except Exception as e:
                self.preview_text.setText(f"Error loading image: {str(e)}")
                self.preview_image_label.setVisible(False)
        
        # Handle PDF files
        elif file_ext == '.pdf':
            # Generate thumbnail for PDF first page
            try:
                pdf_doc = QPdfDocument(self)
                pdf_doc.load(file_path)
                # Render first page as thumbnail if available
                if pdf_doc.pageCount() > 0:
                    img = pdf_doc.render(0, QSize(100, 100))
                    pix = QPixmap.fromImage(img)
                    thumb = pix.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.preview_image_label.setPixmap(thumb)
                    self.preview_image_label.setVisible(True)
                else:
                    self.preview_image_label.setVisible(False)
                pdf_doc.close()
            except Exception:
                self.preview_image_label.setVisible(False)
             
            # Use stored extracted text if available, otherwise extract (fallback)
            if extracted_text is None:
                extracted_text = self.document_processor.extract_text(file_path)
            
            file_size = os.path.getsize(file_path) / 1024  # KB
            preview_text = (
                f"PDF Preview: {os.path.basename(file_path)}\n"
                f"Size: {file_size:.1f} KB\n\n"
                f"Extracted Text (first 5000 characters):\n"
                f"{extracted_text[:5000]}{'...' if len(extracted_text) > 5000 else ''}"
            )
            self.preview_text.setText(preview_text)
          # Handle Word documents (.doc and .docx)
        elif file_ext in ['.doc', '.docx']:
            # Show generic file icon as thumbnail for Word docs
            try:
                style = self.style()
                if style is not None:
                    icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
                    pixmap = icon.pixmap(100, 100)
                    self.preview_image_label.setPixmap(pixmap)
                    self.preview_image_label.setVisible(True)
                else:
                    self.preview_image_label.setVisible(False)
            except Exception:
                self.preview_image_label.setVisible(False)
            try:
                # Use stored extracted text if available, otherwise extract (fallback)
                if extracted_text is None:
                    extracted_text = self.document_processor.extract_text(file_path)
                    
                file_size = os.path.getsize(file_path) / 1024  # KB
                preview_text = (
                    f"DOC Preview: {os.path.basename(file_path)}\n"
                    f"Size: {file_size:.1f} KB\n\n"
                    f"Extracted Text (first 5000 chars):\n"
                    f"{extracted_text[:5000]}{'...' if len(extracted_text) > 5000 else ''}"
                )
                self.preview_text.setText(preview_text)
            except Exception as e:
                self.preview_text.setText(f"Error previewing document: {str(e)}")
        
        # Handle other file types
        else:
            self.preview_image_label.setVisible(False)
            self.preview_text.setText(f"No preview available for {file_ext} files.")
    
    def add_file_to_model(self, analysis_data):
        """Add a file analysis result to the model"""
        row = []
        
        # Checkbox
        checkbox_item = QStandardItem()
        checkbox_item.setCheckable(True)
        checkbox_item.setCheckState(Qt.CheckState.Checked)
        row.append(checkbox_item)
        
        # Original filename
        original_name = os.path.basename(analysis_data['original_path'])
        row.append(QStandardItem(original_name))
        
        # New filename
        row.append(QStandardItem(analysis_data['new_filename']))
        
        # Destination folder
        row.append(QStandardItem(analysis_data['destination_folder']))
        
        # Identity
        row.append(QStandardItem(analysis_data['identity']))
        
        # Date
        row.append(QStandardItem(analysis_data['date']))
        
        # Description
        row.append(QStandardItem(analysis_data['description']))
        
        self.file_model.appendRow(row)
    
    def show_context_menu(self, position):
        """Show context menu for file list"""
        index = self.file_view.indexAt(position)
        menu = QMenu(self)
        # Undo actions available globally
        undo_last = QAction("Undo Last Action", self)
        undo_last.triggered.connect(self.undo_last_action_gui)
        menu.addAction(undo_last)
        undo_all = QAction("Undo All Actions", self)
        undo_all.triggered.connect(self.undo_all_actions_gui)
        menu.addAction(undo_all)
        # Row-specific edit
        if index.isValid():
            edit_action = QAction("Edit", self)
            edit_action.triggered.connect(lambda: self.edit_cell(index))
            menu.addAction(edit_action)
        # Show context menu
        viewport = self.file_view.viewport()
        if viewport is not None:
            menu.exec(viewport.mapToGlobal(position))

    def edit_cell(self, index):
        """Edit a cell in the file list"""
        if not index.isValid():
            return
        
        # Map proxy index to source index for editing
        source_index = self.proxy_model.mapToSource(index)
        item = self.file_model.itemFromIndex(source_index)
        if not item:
            return
        
        current_text = item.text()
        new_text, ok = QInputDialog.getText(self, "Edit", "Enter new value:", text=current_text)
        
        if ok and new_text != current_text:
            item.setText(new_text)
            
            # Update the corresponding analysis data
            row = source_index.row()
            column = source_index.column()
            
            if row < len(self.analyzed_files):                
                if column == 2:  # New filename
                    self.analyzed_files[row]['new_filename'] = new_text
                elif column == 3:  # Destination folder
                    self.analyzed_files[row]['destination_folder'] = new_text
                elif column == 4:  # Identity
                    self.analyzed_files[row]['identity'] = new_text
                elif column == 5:  # Date
                    self.analyzed_files[row]['date'] = new_text
                elif column == 6:  # Description
                    self.analyzed_files[row]['description'] = new_text

    def select_all_files(self):
        """Select all files in the list"""
        for row in range(self.file_model.rowCount()):
            index = self.file_model.index(row, 0)
            self.file_model.setData(index, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)

    def unselect_all_files(self):
        """Unselect all files in the list"""
        for row in range(self.file_model.rowCount()):
            index = self.file_model.index(row, 0)
            self.file_model.setData(index, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    
    def process_files(self):
        """Process the selected files (move and rename)"""
        selected_files = []
        
        for row in range(self.file_model.rowCount()):
            # Check checkbox state using the same method as the custom delegate
            index = self.file_model.index(row, 0)
            checkbox_state = self.file_model.data(index, Qt.ItemDataRole.CheckStateRole)
            
            if checkbox_state == Qt.CheckState.Checked:
                if row < len(self.analyzed_files):
                    # Get updated values from the model
                    new_filename_item = self.file_model.item(row, 2)
                    destination_folder_item = self.file_model.item(row, 3)
                    identity_item = self.file_model.item(row, 4)
                    date_item = self.file_model.item(row, 5)
                    description_item = self.file_model.item(row, 6)

                    new_filename = new_filename_item.text() if new_filename_item else ""
                    destination_folder = destination_folder_item.text() if destination_folder_item else ""
                    identity = identity_item.text() if identity_item else ""
                    date = date_item.text() if date_item else ""
                    description = description_item.text() if description_item else ""
                    
                    file_data = self.analyzed_files[row].copy()
                    file_data.update({
                        'new_filename': new_filename,
                        'destination_folder': destination_folder,
                        'identity': identity,
                        'date': date,
                        'description': description
                    })
                    selected_files.append(file_data)
                           
        if not selected_files:
            QMessageBox.warning(self, "No Files Selected", "Please select files to process.")
            return
        
        # Create progress dialog
        progress = QProgressDialog("Processing files...", "Cancel", 0, len(selected_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        successful_moves = 0
        failed_moves = []
        source_folders = set()  # Track source folders for cleanup
        
        for i, file_data in enumerate(selected_files):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(f"Processing: {file_data['new_filename']}")
            QApplication.processEvents()
            
            try:
                # Construct full destination path
                destination_path = os.path.join(self.file_handler.base_output_dir, file_data['destination_folder'])
                  # Move and rename file                # Create analysis result dictionary from the updated data
                analysis_result = {
                    'identity': file_data['identity'],
                    'date': file_data['date'],
                    'description': file_data['description'],
                    'category': file_data.get('category', 'Other')  # Default to Other if not set
                }
                
                result = self.file_handler.move_and_rename_file(
                    file_data['original_path'],
                    file_data['new_filename'],
                    destination_path,
                    analysis_result,
                    ""  # extracted_text - empty string as fallback
                )
                
                if result:  # Success returns the new path
                    successful_moves += 1
                    # Track the source folder for cleanup
                    source_folder = os.path.dirname(file_data['original_path'])
                    source_folders.add(source_folder)
                else:  # Failure returns None
                    failed_moves.append(f"{file_data['new_filename']}: Failed to move file")
                    
            except Exception as e:
                error_msg = str(e)
                # Provide more user-friendly error messages for common issues
                if "being used by another process" in error_msg:
                    error_msg = "File is open in another application. Please close all PDF viewers and try again."
                elif "Access is denied" in error_msg:
                    error_msg = "Access denied. File may be read-only or you may need administrator permissions."
                elif "Failed to move file after" in error_msg and "attempts" in error_msg:
                    error_msg = "File is locked by another process. Please close any applications using this file."
                
                failed_moves.append(f"{file_data['new_filename']}: {error_msg}")
        
        progress.setValue(len(selected_files))
        progress.close()
        
        # Show results
        message = f"Processing complete!\n\nSuccessful: {successful_moves}\nFailed: {len(failed_moves)}"
        if failed_moves:
            message += f"\n\nFailed files:\n" + "\n".join(failed_moves[:5])
            if len(failed_moves) > 5:
                message += f"\n... and {len(failed_moves) - 5} more"
        
        # Clean up empty source folders after successful processing
        if successful_moves > 0 and source_folders:
            logging.info(f"Cleaning up {len(source_folders)} source folders after processing {successful_moves} files")
            deleted_folders = self.file_handler.cleanup_empty_folders(list(source_folders))
            if deleted_folders:
                message += f"\n\nEmpty folders cleaned up: {len(deleted_folders)}"
                logging.info(f"Deleted {len(deleted_folders)} empty folders: {deleted_folders}")
        
        QMessageBox.information(self, "Processing Results", message)
        
        # Clear the results after successful processing
        if successful_moves > 0:
            self.file_model.clear()
            self.file_model.setHorizontalHeaderLabels([
                "✓", "Original Filename", "New Filename", "Destination Folder", 
                "Identity", "Date", "Description"
            ])
            self.analyzed_files = []
            # Refresh source folder view after moving files and deleting empty folders
            if self.current_folder:
                self.file_system_model.setRootPath(self.current_folder)
                self.file_tree.setRootIndex(self.file_system_model.index(self.current_folder))

    def closeEvent(self, event):
        """Save configuration when the application is closing."""
        # Stop analysis thread if running
        self._cleanup_analysis_thread()
        
        # Remove our logging handler to prevent issues
        if hasattr(self, 'qt_log_handler'):
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.qt_log_handler)
        
        self.save_config()
        event.accept()

    def _update_progress_label(self):
        """Update and elide the progress filename label to fit available width."""
        try:
            if not hasattr(self, 'progress_label') or self.progress_label is None:
                return
            text = getattr(self, '_current_progress_filename', None)
            if not text:
                self.progress_label.setText("")
                return
            prefix = "Analyzing: "
            full = prefix + text
            fm = self.progress_label.fontMetrics() if hasattr(self.progress_label, 'fontMetrics') else QFontMetrics(self.font())
            width = self.progress_label.width()
            if width <= 0:
                self.progress_label.setText(full)
                return
            elided = fm.elidedText(full, Qt.TextElideMode.ElideMiddle, width - 8)
            self.progress_label.setText(elided)
        except Exception:
            # Fallback: set raw text
            self.progress_label.setText(f"Analyzing: {getattr(self, '_current_progress_filename', '')}")

    def resizeEvent(self, event):
        """Ensure progress label stays elided correctly on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'progress_label') and self.progress_label.isVisible():
            self._update_progress_label()

    def set_default_source_folder(self, folder_path):
        """Set the default source folder and initialize the file browser"""
        if folder_path and os.path.exists(folder_path):
            self.current_folder = folder_path
            self.source_folder_edit.setText(folder_path)
            
            # Set up file system model
            self.file_system_model.setRootPath(folder_path)
            self.file_tree.setRootIndex(self.file_system_model.index(folder_path))
            
            # Enable action buttons
            self.analyze_selected_btn.setEnabled(True)
            self.analyze_all_btn.setEnabled(True)
            
            # Hide unnecessary columns
            self.file_tree.hideColumn(1)  # Size
            self.file_tree.hideColumn(2)  # Type
            self.file_tree.hideColumn(3)  # Date Modified

    def undo_last_action_gui(self):
        """Handle undo of the last file action via GUI"""
        try:
            result = self.file_handler.undo_last_action()
            if result:
                QMessageBox.information(self, "Undo Last", "Last action undone successfully.")
            else:
                QMessageBox.warning(self, "Undo Last", "Could not undo the last action. Either no actions to undo or the file could not be restored.")
        except Exception as e:
            logging.error(f"Error in undo_last_action_gui: {str(e)}")
            QMessageBox.critical(self, "Undo Error", f"An error occurred while undoing the last action:\n{str(e)}")
    
    def undo_all_actions_gui(self):
        """Handle undo of all file actions via GUI"""
        try:
            # Load current actions to check if there are any
            actions = self.file_handler._load_actions()
            if not actions:
                QMessageBox.information(self, "Undo All", "No actions to undo.")
                return
            
            # Confirm with user since this could be a big operation
            reply = QMessageBox.question(
                self, 
                "Confirm Undo All", 
                f"Are you sure you want to undo all {len(actions)} file actions?\n\n"
                "This will attempt to restore all processed files to their original locations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            result = self.file_handler.undo_all_actions()
            
            # Check how many actions are left (failed undos)
            remaining_actions = self.file_handler._load_actions()
            
            if result:
                if remaining_actions:
                    # Partial success
                    successful_count = len(actions) - len(remaining_actions)
                    QMessageBox.warning(
                        self, 
                        "Undo All - Partial Success", 
                        f"Undo partially completed:\n"
                        f"• Successfully undone: {successful_count} actions\n"
                        f"• Failed to undo: {len(remaining_actions)} actions\n\n"
                        "Some files may have been moved or deleted manually. "
                        "Check the logs for more details."
                    )
                else:
                    # Complete success
                    QMessageBox.information(self, "Undo All", "All actions undone successfully.")
            else:
                QMessageBox.warning(
                    self, 
                    "Undo All Failed", 
                    "Could not undo any actions. Files may have been moved or deleted manually. "
                    "Check the logs for more details."
                )
        except Exception as e:
            logging.error(f"Error in undo_all_actions_gui: {str(e)}")
            QMessageBox.critical(self, "Undo Error", f"An error occurred while undoing actions:\n{str(e)}")

def main():
    """Main function to run the GUI application"""
    app = QApplication(sys.argv)
    
    # Import the required modules
    try:
        from document_processor import DocumentProcessor
        from llm_analyzer import LLMAnalyzer
        from file_handler import FileHandler
        
        # Initialize components
        document_processor = DocumentProcessor()
        llm_analyzer = LLMAnalyzer()
        file_handler = FileHandler(base_output_dir=r"E:\scanned documents")
        
        # Create and show GUI
        window = FileOrganizerGUI(document_processor, llm_analyzer, file_handler)
        window.show()
        
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
