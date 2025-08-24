import os
import shutil
from datetime import datetime
import logging
from PIL import Image, ExifTags
import json
import uuid
import time
import psutil

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
        categories = [
            "Medical", "Identification", "Home", "Auto", "SysAdmin",
            "School", "Cooking", "Financial", "Travel", "Employment",
            "Photography", "Hobbies", "Memories", "Other"
        ]
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
    
    def _is_file_in_use(self, file_path):
        """Check if a file is currently in use by another process."""
        try:
            # Try to open the file in exclusive mode
            with open(file_path, 'r+b'):
                pass
            return False
        except (OSError, IOError):
            return True
    
    def _get_processes_using_file(self, file_path):
        """Get list of processes currently using the file."""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for f in proc.open_files():
                        if f.path.lower() == file_path.lower():
                            processes.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            logging.debug(f"Could not check processes using file: {e}")
        return processes
    
    def _wait_for_file_release(self, file_path, max_wait_time=30, check_interval=1):
        """Wait for a file to be released by other processes."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if not self._is_file_in_use(file_path):
                return True
            
            # Log which processes are using the file (for debugging)
            if time.time() - start_time > 5:  # Only log after 5 seconds
                processes = self._get_processes_using_file(file_path)
                if processes:
                    logging.info(f"File {file_path} is being used by: {', '.join(processes)}")
                else:
                    logging.info(f"File {file_path} appears to be locked but couldn't identify the process")
            
            logging.debug(f"Waiting for file to be released: {file_path} ({time.time() - start_time:.1f}s)")
            time.sleep(check_interval)
        
        return False
    
    def move_and_rename_file(self, original_path, new_filename, destination_path, analysis_result, extracted_text):
        """Move and rename the file, adding metadata if appropriate."""
        try:
            logging.info(f"Moving file from {original_path} to {destination_path}/{new_filename}")
            
            # Validate input paths
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"Source file does not exist: {original_path}")
            
            # Check if file is currently in use
            if self._is_file_in_use(original_path):
                logging.warning(f"File is currently in use: {original_path}")
                processes = self._get_processes_using_file(original_path)
                if processes:
                    logging.warning(f"Processes using the file: {', '.join(processes)}")
                
                logging.info(f"Waiting for file to be released (up to 30 seconds): {os.path.basename(original_path)}")
                if not self._wait_for_file_release(original_path, max_wait_time=30):
                    raise PermissionError(f"File is still in use after 30 seconds: {original_path}. "
                                       f"Please close any applications that might have the file open and try again.")
            
            # Ensure destination directory exists
            if not os.path.exists(destination_path):
                logging.info(f"Creating destination directory: {destination_path}")
                os.makedirs(destination_path, exist_ok=True)
            
            # Create the full destination path
            new_path = os.path.join(destination_path, new_filename)
            logging.debug(f"Full destination path: {new_path}")
            
            # Check if destination file already exists
            counter = 1
            base_name, ext = os.path.splitext(new_filename)
            while os.path.exists(new_path):
                new_path = os.path.join(destination_path, f"{base_name} ({counter}){ext}")
                counter += 1
            
            # Backup original file before moving (with retry for locked files)
            backup_name = f"{uuid.uuid4()}_{os.path.basename(original_path)}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            logging.debug(f"Creating backup at: {backup_path}")
            
            # Retry backup creation if file is temporarily locked
            max_backup_retries = 3
            for backup_attempt in range(max_backup_retries):
                try:
                    shutil.copy2(original_path, backup_path)
                    break
                except (OSError, IOError) as e:
                    if backup_attempt < max_backup_retries - 1:
                        logging.warning(f"Backup attempt {backup_attempt + 1} failed, retrying in 2 seconds: {e}")
                        time.sleep(2)
                    else:
                        raise Exception(f"Failed to create backup after {max_backup_retries} attempts: {e}")

            # Move the file to new location (with retry for locked files)
            logging.debug(f"Moving file from {original_path} to {new_path}")
            
            max_move_retries = 5
            for move_attempt in range(max_move_retries):
                try:
                    shutil.move(original_path, new_path)
                    break
                except (OSError, IOError) as e:
                    if "being used by another process" in str(e) or "Access is denied" in str(e):
                        if move_attempt < max_move_retries - 1:
                            wait_time = (move_attempt + 1) * 2  # Exponential backoff: 2, 4, 6, 8 seconds
                            logging.warning(f"Move attempt {move_attempt + 1} failed (file in use), retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            
                            # Check again if file is still in use and wait
                            if self._is_file_in_use(original_path):
                                if not self._wait_for_file_release(original_path, max_wait_time=10):
                                    logging.warning(f"File still in use after waiting, but continuing with retry...")
                        else:
                            # Final attempt failed, clean up backup and re-raise
                            if os.path.exists(backup_path):
                                try:
                                    os.remove(backup_path)
                                except:
                                    pass
                            raise Exception(f"Failed to move file after {max_move_retries} attempts. File may still be open in another application. "
                                          f"Please close any PDF viewers or other applications that might have the file open and try again. "
                                          f"Original error: {e}")
                    else:
                        # Different type of error, don't retry
                        if os.path.exists(backup_path):
                            try:
                                os.remove(backup_path)
                            except:
                                pass
                        raise
            
            # Log the action for possible undo
            self._log_action(original_path, new_path, backup_path)
            
            # Add metadata to image files
            file_ext = os.path.splitext(original_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                self.add_metadata_to_image(new_path, analysis_result, extracted_text)

            # Set the file's modification date
            try:
                target_date_str = analysis_result.get("date")
                target_timestamp = None
                
                if target_date_str:
                    try:
                        # Attempt to parse the date from LLM analysis
                        dt_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
                        target_timestamp = dt_obj.timestamp()
                    except ValueError:
                        logging.warning(f"Could not parse date '{target_date_str}' from analysis. Falling back to file creation date.")
                
                if target_timestamp is None:
                    # Fallback to original file's creation date (scan date)
                    creation_time = self.get_file_creation_date(backup_path) # Use backup to get original metadata
                    target_timestamp = creation_time.timestamp()

                # Set access and modification times
                os.utime(new_path, (target_timestamp, target_timestamp))
                logging.info(f"Set modification date for {new_path} to {datetime.fromtimestamp(target_timestamp)}")

            except Exception as e:
                logging.error(f"Failed to set modification date for {new_path}: {str(e)}")
            
            logging.info(f"File moved and renamed successfully: {new_path}")
            
            return new_path
        except Exception as e:
            logging.error(f"Error moving file {original_path}: {str(e)}")
            logging.error(f"Destination path was: {destination_path}")
            logging.error(f"New filename was: {new_filename}")
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
        
        try:
            # Move file back to original location if it exists at new location
            if os.path.exists(action['new_path']):
                # Ensure the destination directory exists
                original_dir = os.path.dirname(action['original_path'])
                if not os.path.exists(original_dir):
                    logging.info(f"Creating directory for undo: {original_dir}")
                    os.makedirs(original_dir, exist_ok=True)
                
                # Move file back
                shutil.move(action['new_path'], action['original_path'])
                logging.info(f"Undone action: moved {action['new_path']} back to {action['original_path']}")
            else:
                # File doesn't exist at new location, try to restore from backup
                if os.path.exists(action['backup_path']):
                    # Ensure the destination directory exists
                    original_dir = os.path.dirname(action['original_path'])
                    if not os.path.exists(original_dir):
                        logging.info(f"Creating directory for backup restore: {original_dir}")
                        os.makedirs(original_dir, exist_ok=True)
                    
                    shutil.copy2(action['backup_path'], action['original_path'])
                    logging.info(f"Restored from backup: {action['backup_path']} to {action['original_path']}")
                else:
                    logging.warning(f"Cannot undo action: neither new path {action['new_path']} nor backup {action['backup_path']} exists")
            
            # Remove backup if it exists
            if os.path.exists(action['backup_path']):
                os.remove(action['backup_path'])
                logging.info(f"Removed backup file: {action['backup_path']}")
            
            # Save updated actions list
            self._save_actions(actions)
            return True
            
        except Exception as e:
            logging.error(f"Error undoing action: {str(e)}")
            logging.error(f"Action details: new_path={action['new_path']}, original_path={action['original_path']}, backup_path={action['backup_path']}")
            # Re-add the action since we couldn't undo it
            actions.append(action)
            self._save_actions(actions)
            return False

    def undo_all_actions(self):
        """Undo all recorded move and rename actions."""
        actions = self._load_actions()
        if not actions:
            logging.info("No actions to undo.")
            return False
        
        successfully_undone = []
        failed_undos = []
        
        # Undo in reverse order
        for action in reversed(actions):
            try:
                # Move file back to original location if it exists at new location
                if os.path.exists(action['new_path']):
                    # Ensure the destination directory exists
                    original_dir = os.path.dirname(action['original_path'])
                    if not os.path.exists(original_dir):
                        logging.info(f"Creating directory for undo: {original_dir}")
                        os.makedirs(original_dir, exist_ok=True)
                    
                    # Move file back
                    shutil.move(action['new_path'], action['original_path'])
                    logging.info(f"Undone action: moved {action['new_path']} back to {action['original_path']}")
                    successfully_undone.append(action)
                else:
                    # File doesn't exist at new location, try to restore from backup
                    if os.path.exists(action['backup_path']):
                        # Ensure the destination directory exists
                        original_dir = os.path.dirname(action['original_path'])
                        if not os.path.exists(original_dir):
                            logging.info(f"Creating directory for backup restore: {original_dir}")
                            os.makedirs(original_dir, exist_ok=True)
                        
                        shutil.copy2(action['backup_path'], action['original_path'])
                        logging.info(f"Restored from backup: {action['backup_path']} to {action['original_path']}")
                        successfully_undone.append(action)
                    else:
                        logging.warning(f"Cannot undo action: neither new path {action['new_path']} nor backup {action['backup_path']} exists")
                        failed_undos.append(action)
                        continue
                
                # Remove backup if it exists (only for successfully undone actions)
                if os.path.exists(action['backup_path']):
                    os.remove(action['backup_path'])
                    logging.info(f"Removed backup file: {action['backup_path']}")
                    
            except Exception as e:
                logging.error(f"Error undoing action: {str(e)}")
                logging.error(f"Action details: new_path={action['new_path']}, original_path={action['original_path']}, backup_path={action['backup_path']}")
                failed_undos.append(action)
        
        # Update the actions log to keep only the failed actions
        self._save_actions(failed_undos)
        
        # Report results
        if successfully_undone:
            logging.info(f"Successfully undone {len(successfully_undone)} actions")
        if failed_undos:
            logging.warning(f"Failed to undo {len(failed_undos)} actions")
        
        # Return True if we undid at least some actions
        return len(successfully_undone) > 0
    
    def delete_empty_folder(self, folder_path):
        """Safely delete a folder if it's empty."""
        try:
            if not os.path.exists(folder_path):
                logging.debug(f"Folder does not exist: {folder_path}")
                return False
                
            if not os.path.isdir(folder_path):
                logging.debug(f"Path is not a directory: {folder_path}")
                return False
            
            # Check if folder is empty
            if os.listdir(folder_path):
                logging.debug(f"Folder is not empty: {folder_path}")
                return False
            
            # Delete the empty folder
            os.rmdir(folder_path)
            logging.info(f"Deleted empty folder: {folder_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting empty folder {folder_path}: {str(e)}")
            return False
    
    def cleanup_empty_folders(self, folder_paths):
        """Clean up multiple folders that may be empty after file processing."""
        deleted_folders = []
        
        # Sort paths by depth (deepest first) to ensure we delete child folders before parent folders
        sorted_paths = sorted(set(folder_paths), key=lambda p: p.count(os.sep), reverse=True)
        
        for folder_path in sorted_paths:
            if self.delete_empty_folder(folder_path):
                deleted_folders.append(folder_path)
                
                # Also try to delete parent folders if they become empty
                parent_folder = os.path.dirname(folder_path)
                while parent_folder and parent_folder != folder_path:
                    if self.delete_empty_folder(parent_folder):
                        deleted_folders.append(parent_folder)
                        folder_path = parent_folder
                        parent_folder = os.path.dirname(folder_path)
                    else:
                        break
        
        if deleted_folders:
            logging.info(f"Cleanup complete: deleted {len(deleted_folders)} empty folders")
        else:
            logging.info("No empty folders were found to delete")
            
        return deleted_folders
