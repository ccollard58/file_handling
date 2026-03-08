import sys
import os
import json
from PyQt6.QtWidgets import QApplication
from document_processor import DocumentProcessor
from llm_analyzer import LLMAnalyzer
from file_handler import FileHandler
from gui_simplified import FileOrganizerGUI
import logging
from logging.handlers import RotatingFileHandler


def load_saved_llm_settings() -> dict:
    """Load persisted LLM settings from the user config file if available."""
    config_file = os.path.join(os.path.expanduser("~"), ".document_organizer_config.json")
    default_settings = {
        "provider": "ollama",
        "model": "gemma3:latest",
        "vision_model": "llava:latest",
        "temperature": 0.6,
        "google_api_key": None,
    }

    if not os.path.exists(config_file):
        return default_settings

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = json.load(file)
        llm_settings = config.get("llm_settings", {})
        return {
            "provider": llm_settings.get("provider", default_settings["provider"]),
            "model": llm_settings.get("model", default_settings["model"]),
            "vision_model": llm_settings.get("vision_model", default_settings["vision_model"]),
            "temperature": llm_settings.get("temperature", default_settings["temperature"]),
            "google_api_key": llm_settings.get("google_api_key", "") or None,
        }
    except Exception as exc:
        logging.warning(f"Unable to load LLM settings from config: {exc}")
        return default_settings

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
        saved_llm_settings = load_saved_llm_settings()

        # Initialize components
        llm_analyzer = LLMAnalyzer(
            model=saved_llm_settings["model"],
            temperature=saved_llm_settings["temperature"],
            vision_model=saved_llm_settings["vision_model"],
            provider=saved_llm_settings["provider"],
            google_api_key=saved_llm_settings["google_api_key"],
        )
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
