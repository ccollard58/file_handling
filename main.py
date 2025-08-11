import sys
import os
from PyQt6.QtWidgets import QApplication
from document_processor import DocumentProcessor
from llm_analyzer import LLMAnalyzer
from file_handler import FileHandler
from gui_simplified import FileOrganizerGUI
import logging
from logging.handlers import RotatingFileHandler
import os

def main():
    # Set up logging with rotating file handler
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Set up rotating file handler (10MB max, keep 5 backup files)
    log_file = os.path.join(logs_dir, "document_organizer.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Log startup
    logging.info("Starting Document Organizer application")
    logging.info(f"Logging to file: {os.path.abspath(log_file)}")
    
    # Initialize application
    app = QApplication(sys.argv)
    app.setApplicationName("Document Organizer")
    
    try:
        # Initialize components
        llm_analyzer = LLMAnalyzer(vision_model="llava:latest")  # Set default vision model
        document_processor = DocumentProcessor(llm_analyzer)  # Pass llm_analyzer to document processor
        
        # Default output directory is the user's Documents folder
        # For testing, use e:\junk as the output directory
        default_output_dir = r"E:\scanned documents"
        file_handler = FileHandler(default_output_dir)
        
        # Create and show GUI
        gui = FileOrganizerGUI(document_processor, llm_analyzer, file_handler)
        gui.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
