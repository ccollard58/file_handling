#!/usr/bin/env python3
"""
File Lock Checker - Utility to check which files in a directory are currently locked.
This can help users identify which files might cause issues during processing.
"""

import os
import sys
import argparse
import psutil
from pathlib import Path

def is_file_locked(file_path):
    """Check if a file is currently locked by another process."""
    try:
        with open(file_path, 'r+b'):
            pass
        return False
    except (OSError, IOError):
        return True

def get_processes_using_file(file_path):
    """Get list of processes currently using the file."""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for f in proc.open_files():
                    if f.path.lower() == file_path.lower():
                        processes.append({
                            'name': proc.info['name'],
                            'pid': proc.info['pid']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass
    return processes

def check_directory(directory_path, extensions=None):
    """Check all files in a directory for locks."""
    if extensions is None:
        extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.docx', '.doc', '.xlsx']
    
    locked_files = []
    unlocked_files = []
    
    print(f"Checking files in: {directory_path}")
    print("=" * 60)
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                if is_file_locked(file_path):
                    processes = get_processes_using_file(file_path)
                    locked_files.append({
                        'path': file_path,
                        'processes': processes
                    })
                    
                    rel_path = os.path.relpath(file_path, directory_path)
                    print(f"🔒 LOCKED: {rel_path}")
                    
                    if processes:
                        for proc in processes:
                            print(f"   └─ Used by: {proc['name']} (PID: {proc['pid']})")
                    else:
                        print(f"   └─ Locked by unknown process")
                    print()
                else:
                    unlocked_files.append(file_path)
    
    return locked_files, unlocked_files

def main():
    parser = argparse.ArgumentParser(
        description="Check which files in a directory are currently locked by other processes"
    )
    parser.add_argument(
        "directory", 
        help="Directory path to check for locked files"
    )
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.docx', '.doc', '.xlsx'],
        help="File extensions to check (default: common document types)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show locked files, not summary"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.directory):
        print(f"Error: Directory does not exist: {args.directory}")
        sys.exit(1)
    
    if not os.path.isdir(args.directory):
        print(f"Error: Path is not a directory: {args.directory}")
        sys.exit(1)
    
    try:
        # Check files
        locked_files, unlocked_files = check_directory(args.directory, args.extensions)
        
        if not args.quiet:
            # Print summary
            print("=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total files checked: {len(locked_files) + len(unlocked_files)}")
            print(f"Locked files: {len(locked_files)}")
            print(f"Available files: {len(unlocked_files)}")
            
            if locked_files:
                print("\n⚠️  RECOMMENDATIONS:")
                print("   - Close any PDF viewers (Adobe Reader, Sumatra PDF, etc.)")
                print("   - Wait for Windows Search Indexer to complete")
                print("   - Check if antivirus is scanning files")
                print("   - Use Task Manager to end processes if necessary")
                print("\n   Re-run this tool after closing applications to verify files are unlocked.")
            else:
                print("\n✅ All files are available for processing!")
        
        # Exit with appropriate code
        sys.exit(1 if locked_files else 0)
        
    except Exception as e:
        print(f"Error checking files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()