import os
import shutil
from datetime import datetime
import logging
from PIL import Image, ExifTags
import json
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileHandler:
    def __init__(self, base_output_dir):
        """Initialize with the base output directory."""
        self.base_output_dir = base_output_dir
        
        self.backup_dir = os.path.join(self.base_output_dir, '.backup')
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        self.actions_log_path = os.path.join(self.base_output_dir, 'file_actions_log.json')
        if not os.path.exists(self.actions_log_path):
            with open(self.actions_log_path, 'w') as f:
                json.dump([], f)
        
        self._ensure_directories()
        
    def _ensure_directories(self):    
        """Ensure all necessary directories exist."""
        categories = ["Medical", "Receipts", "Contracts", "Photographs", "Other"]
        for category in categories:
            dir_path = os.path.join(self.base_output_dir, category)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Created directory: {dir_path}")
    
    def get_file_creation_date(self, file_path):
        """Get file creation date from filesystem."""
        try:
            # Get file creation time on Windows
            ctime = os.path.getctime(file_path)
            return datetime.fromtimestamp(ctime)
        except Exception as e:
            logging.error(f"Error getting creation date for {file_path}: {str(e)}")
            return datetime.now()
    
    def generate_new_filename(self, analysis_result, original_filename):
        """Generate new filename based on analysis results."""
        # Extract components
        identity = analysis_result["identity"]
        date = analysis_result["date"]
        description = analysis_result["description"]
        
        # Get file extension
        _, ext = os.path.splitext(original_filename)
        
        # Generate new filename, omitting identity if it's Unknown
        if identity and identity.lower() != "unknown":
            new_filename = f"{identity} {date} - {description}{ext}"
        else:
            new_filename = f"{date} - {description}{ext}"
        
        # Replace invalid characters
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            new_filename = new_filename.replace(char, '_')
        
        return new_filename
    def get_destination_path(self, analysis_result):
        """Determine the destination path based on document category."""
        # Use LLM-defined categories directly; default to Other if unknown
        category = analysis_result["category"]
        valid_categories = [
            "Medical", "Identification", "Home", "Auto", "SysAdmin",
            "School", "Cooking", "Financial", "Travel", "Employment",
            "Photography", "Hobbies", "Memories", "Other"
        ]
        if category not in valid_categories:
            category = "Other"
        return os.path.join(self.base_output_dir, category)
    
    def add_metadata_to_image(self, file_path, analysis_result, extracted_text):
        """Add metadata to image files."""
        # Only process image files
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg']:
            return
        
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Create a dictionary of metadata
            metadata = {
                "DocumentType": analysis_result["category"],
                "Description": analysis_result["description"],
                "ProcessedDate": datetime.now().strftime("%Y-%m-%d"),
                "Identity": analysis_result["identity"],
                "DocumentDate": analysis_result["date"]
            }
            
            # Add metadata to EXIF
            exif_dict = img.getexif()
            if not exif_dict:
                exif_dict = Image.Exif()
            
            # Add to UserComment field (37510)
            user_comment = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
            exif_dict[37510] = user_comment.encode('utf-16')
            
            # Save the image with the new metadata
            img.save(file_path, exif=exif_dict)
            logging.info(f"Added metadata to {file_path}")
            
        except Exception as e:
            logging.error(f"Error adding metadata to {file_path}: {str(e)}")
    
    def move_and_rename_file(self, original_path, new_filename, destination_path, analysis_result, extracted_text):
        """Move and rename the file, adding metadata if appropriate."""
        try:
            # Ensure destination directory exists
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            
            # Create the full destination path
            new_path = os.path.join(destination_path, new_filename)
            
            # Check if destination file already exists
            counter = 1
            base_name, ext = os.path.splitext(new_filename)
            while os.path.exists(new_path):
                new_path = os.path.join(destination_path, f"{base_name} ({counter}){ext}")
                counter += 1
            
            # Backup original file before moving
            backup_name = f"{uuid.uuid4()}_{os.path.basename(original_path)}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(original_path, backup_path)

            # Copy (move) the file to new location
            shutil.move(original_path, new_path)
            
            # Log the action for possible undo
            self._log_action(original_path, new_path, backup_path)
            
            # Add metadata to image files
            file_ext = os.path.splitext(original_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                self.add_metadata_to_image(new_path, analysis_result, extracted_text)
            
            logging.info(f"File moved and renamed: {new_path}")
            
            return new_path
        except Exception as e:
            logging.error(f"Error moving file {original_path}: {str(e)}")
            return None

    def _load_actions(self):
        """Load actions log from file."""
        with open(self.actions_log_path, 'r') as f:
            return json.load(f)

    def _save_actions(self, actions):
        """Save actions log to file."""
        with open(self.actions_log_path, 'w') as f:
            json.dump(actions, f, indent=2)

    def _log_action(self, original, new, backup):
        """Append a move action to the log."""
        actions = self._load_actions()
        actions.append({'original_path': original, 'new_path': new, 'backup_path': backup})
        self._save_actions(actions)

    def undo_last_action(self):
        """Undo the most recent move and rename action."""
        actions = self._load_actions()
        if not actions:
            logging.info("No actions to undo.")
            return False
        action = actions.pop()
        # Move file back to original location
        if os.path.exists(action['new_path']):
            shutil.move(action['new_path'], action['original_path'])
        # Remove backup once restored
        if os.path.exists(action['backup_path']):
            os.remove(action['backup_path'])
        self._save_actions(actions)
        logging.info(f"Undone action: moved {action['new_path']} back to {action['original_path']}")
        return True

    def undo_all_actions(self):
        """Undo all recorded move and rename actions."""
        actions = self._load_actions()
        if not actions:
            logging.info("No actions to undo.")
            return False
        # Undo in reverse order
        for action in reversed(actions):
            if os.path.exists(action['new_path']):
                shutil.move(action['new_path'], action['original_path'])
            if os.path.exists(action['backup_path']):
                os.remove(action['backup_path'])
            logging.info(f"Undone action: moved {action['new_path']} back to {action['original_path']}")
        # Clear log
        self._save_actions([])
        return True
