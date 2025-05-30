import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QTreeView, QFileDialog, QCheckBox, QComboBox,
                           QMessageBox, QHeaderView, QSplitter, QProgressDialog,
                           QMenu, QInputDialog, QGridLayout, QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt, QFileInfo, QDir, QModelIndex, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFileSystemModel, QAction, QColor, QPixmap, QImage

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

class FileOrganizerGUI(QMainWindow):
    def __init__(self, document_processor, llm_analyzer, file_handler):
        super().__init__()
        self.document_processor = document_processor
        self.llm_analyzer = llm_analyzer
        self.file_handler = file_handler
        self.analyzed_files = []  # Will store analysis results
        self.current_folder = None  # Current selected folder
        
        # Set default output folder
        default_output = r"E:\scanned documents"
        self.file_handler.base_output_dir = default_output
        self.file_handler._ensure_directories()
        
        # Set default source folder to Documents
        # For testing, use E:\Documents as the default source folder
        default_source = r"E:\Documents"
        
        self.init_ui()
        self.output_folder_edit.setText(self.file_handler.base_output_dir)  # Show default folder
        
        # Set the default source folder
        self.set_default_source_folder(default_source)
    
    def init_ui(self):
        """Initialize the simplified user interface"""
        self.setWindowTitle("Document Organizer - Simplified")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Step 1: Output Folder Selection (Compact)
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
        
        # Header with browse button
        folder_header = QHBoxLayout()
        folder_header.addWidget(QLabel("Select Source Folder and Files"))
        folder_header.addStretch()
        
        self.browse_folder_btn = QPushButton("Browse Folder...")
        self.browse_folder_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.browse_folder_btn.clicked.connect(self.browse_source_folder)
        folder_header.addWidget(self.browse_folder_btn)
        
        folder_layout.addLayout(folder_header)
        
        # Current folder status
        self.folder_status_label = QLabel("No folder selected")
        self.folder_status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        folder_layout.addWidget(self.folder_status_label)
        
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
        
        results_layout.addWidget(QLabel("Step 3: Review and Process Results"))
        
        # File list view for results
        self.file_view = QTreeView()
        self.file_model = QStandardItemModel()
        self.file_model.setHorizontalHeaderLabels([
            "✓", "Original Filename", "New Filename", "Destination Folder", 
            "Identity", "Date", "Description"
        ])
        self.file_view.setModel(self.file_model)
        
        # Enable context menu
        self.file_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_view.customContextMenuRequested.connect(self.show_context_menu)
        
        results_layout.addWidget(self.file_view)
        
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
        
        results_layout.addLayout(process_layout)
        
        # Add preview panel
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Image preview label
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumHeight(200)
        self.preview_image_label.setMaximumHeight(200)
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
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Make the new filename column stretch
            
        # Connect selection change to preview update
        self.file_view.selectionModel().selectionChanged.connect(self.update_preview)
    
    def browse_source_folder(self):
        """Browse for source folder and populate file tree"""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.current_folder = folder
            self.folder_status_label.setText(f"Selected: {folder}")
            self.folder_status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
            
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
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)
            self.file_handler.base_output_dir = folder
            self.file_handler._ensure_directories()
    
    def get_selected_files(self):
        """Get list of selected files or folders and return supported files"""
        selected_files = []
        selection_model = self.file_tree.selectionModel()
        if selection_model is None:
            return selected_files
        selection = selection_model.selectedIndexes()
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']
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
        if not self.current_folder:
            return []
        
        all_files = []
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
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
        """Analyze a list of files and populate results"""
        if not file_list:
            return
        
        # Clear previous results
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels([
            "✓", "Original Filename", "New Filename", "Destination Folder", 
            "Identity", "Date", "Description"
        ])
        self.analyzed_files = []
        
        # Create progress dialog
        progress = QProgressDialog("Analyzing files...", "Cancel", 0, len(file_list), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        for i, file_path in enumerate(file_list):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(f"Analyzing: {os.path.basename(file_path)}")
            QApplication.processEvents()
            
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
                    'category': analysis_result['category']
                }
                self.analyzed_files.append(analysis_data)
                
                # Add to model
                self.add_file_to_model(analysis_data)
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
                continue
        
        progress.setValue(len(file_list))
        progress.close()
        
        # Configure view after adding data
        header = self.file_view.header()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            
        # Update preview if available
        self.update_preview()
    
    def update_preview(self):
        """Update the preview panel with the selected file content"""
        if not self.analyzed_files:
            self.preview_text.setText("No files analyzed yet.")
            self.preview_image_label.setVisible(False)
            return
            
        indexes = self.file_view.selectionModel().selectedRows()
        if not indexes:
            self.preview_text.setText("Select a file to see preview.")
            self.preview_image_label.setVisible(False)
            return
            
        row_index = indexes[0].row()
        if 0 <= row_index < len(self.analyzed_files):
            file_path = self.analyzed_files[row_index]['original_path']
            self.show_file_preview(file_path)
    
    def show_file_preview(self, file_path):
        """Show a preview of the selected file"""
        if not os.path.exists(file_path):
            self.preview_text.setText(f"File not found: {file_path}")
            self.preview_image_label.setVisible(False)
            return
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle image files
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
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
                    
                    # Extract text (OCR) for images
                    extracted_text = self.document_processor.extract_text(file_path)
                    
                    preview_text = (
                        f"Image Preview: {os.path.basename(file_path)}\n"
                        f"Dimensions: {dimensions}\n"
                        f"Size: {file_size:.1f} KB\n"
                        f"Format: {file_ext[1:].upper()}\n\n"
                        f"Extracted Text (OCR):\n"
                        f"{extracted_text[:500]}{'...' if len(extracted_text) > 500 else ''}"
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
            self.preview_image_label.setVisible(False)
            
            # Extract text from PDF
            extracted_text = self.document_processor.extract_text(file_path)
            
            file_size = os.path.getsize(file_path) / 1024  # KB
            preview_text = (
                f"PDF Preview: {os.path.basename(file_path)}\n"
                f"Size: {file_size:.1f} KB\n\n"
                f"Extracted Text (first 1000 characters):\n"
                f"{extracted_text[:1000]}{'...' if len(extracted_text) > 1000 else ''}"
            )
            self.preview_text.setText(preview_text)
        
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
        if not index.isValid():
            return
        
        menu = QMenu(self)
        
        edit_action = QAction("Edit", self)
        edit_action.triggered.connect(lambda: self.edit_cell(index))
        menu.addAction(edit_action)
        
        viewport = self.file_view.viewport()
        if viewport is not None:
            menu.exec(viewport.mapToGlobal(position))
    
    def edit_cell(self, index):
        """Edit a cell in the file list"""
        if not index.isValid():
            return
        
        item = self.file_model.itemFromIndex(index)
        if not item:
            return
        
        current_text = item.text()
        new_text, ok = QInputDialog.getText(self, "Edit", "Enter new value:", text=current_text)
        
        if ok and new_text != current_text:
            item.setText(new_text)
            
            # Update the corresponding analysis data
            row = index.row()
            column = index.column()
            
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
            checkbox_item = self.file_model.item(row, 0)
            if checkbox_item:
                checkbox_item.setCheckState(Qt.CheckState.Checked)
    
    def unselect_all_files(self):
        """Unselect all files in the list"""
        for row in range(self.file_model.rowCount()):
            checkbox_item = self.file_model.item(row, 0)
            if checkbox_item:
                checkbox_item.setCheckState(Qt.CheckState.Unchecked)
    
    def process_files(self):
        """Process the selected files (move and rename)"""
        selected_files = []
        
        for row in range(self.file_model.rowCount()):
            checkbox_item = self.file_model.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
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
                else:  # Failure returns None
                    failed_moves.append(f"{file_data['new_filename']}: Failed to move file")
                    
            except Exception as e:
                failed_moves.append(f"{file_data['new_filename']}: {str(e)}")
        
        progress.setValue(len(selected_files))
        progress.close()
        
        # Show results
        message = f"Processing complete!\n\nSuccessful: {successful_moves}\nFailed: {len(failed_moves)}"
        if failed_moves:
            message += f"\n\nFailed files:\n" + "\n".join(failed_moves[:5])
            if len(failed_moves) > 5:
                message += f"\n... and {len(failed_moves) - 5} more"
        
        QMessageBox.information(self, "Processing Results", message)
        
        # Clear the results after successful processing
        if successful_moves > 0:
            self.file_model.clear()
            self.file_model.setHorizontalHeaderLabels([
                "✓", "Original Filename", "New Filename", "Destination Folder", 
                "Identity", "Date", "Description"
            ])
            self.analyzed_files = []
            # Refresh source folder view after moving files
            self.file_system_model.setRootPath(self.current_folder)
            self.file_tree.setRootIndex(self.file_system_model.index(self.current_folder))

    def set_default_source_folder(self, folder_path):
        """Set the default source folder and initialize the file browser"""
        if folder_path and os.path.exists(folder_path):
            self.current_folder = folder_path
            self.folder_status_label.setText(f"Default: {folder_path}")
            self.folder_status_label.setStyleSheet("QLabel { color: #2196F3; font-weight: bold; }")
            
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
