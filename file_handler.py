import os
import shutil
from datetime import datetime
import logging
from PIL import Image, ExifTags

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileHandler:
    def __init__(self, base_output_dir):
        """Initialize with the base output directory."""
        self.base_output_dir = base_output_dir
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
        category = analysis_result["category"]
        # If category doesn't match our predefined ones, default to "Other"
        if category not in ["Medical Documents", "Receipts", "Contracts", "Photographs", "Other"]:
            category = "Other"
        
        # Map full category names to directory names
        category_mapping = {
            "Medical Documents": "Medical",
            "Receipts": "Receipts", 
            "Contracts": "Contracts",
            "Photographs": "Photographs",
            "Other": "Other"
        }
        
        folder_name = category_mapping.get(category, "Other")
        return os.path.join(self.base_output_dir, folder_name)
    
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
            
            # Copy the file to new location
            shutil.copy2(original_path, new_path)
            
            # Add metadata to image files
            file_ext = os.path.splitext(original_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                self.add_metadata_to_image(new_path, analysis_result, extracted_text)
            
            logging.info(f"File moved and renamed: {new_path}")
            
            return new_path
        except Exception as e:
            logging.error(f"Error moving file {original_path}: {str(e)}")
            return None
