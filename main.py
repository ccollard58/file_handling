import sys
import os
from PyQt6.QtWidgets import QApplication
from document_processor import DocumentProcessor
from llm_analyzer import LLMAnalyzer
from file_handler import FileHandler
from gui_simplified import FileOrganizerGUI
import logging

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("document_organizer.log"),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Log startup
    logging.info("Starting Document Organizer application")
    
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
