#!/usr/bin/env python3
"""
Command-line interface for the enhanced document processing system
"""

import argparse
import os
import sys
from pathlib import Path
from document_processor import DocumentProcessor
from llm_analyzer import LLMAnalyzer
import json

def process_single_file(file_path, output_dir=None):
    """Process a single file and extract text"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔍 Processing: {file_path}")
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # Extract text
    extracted_text = doc_processor.extract_text(file_path)
    
    if extracted_text.strip():
        print(f"✅ Successfully extracted {len(extracted_text.strip())} characters")
        
        # Determine output file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"extracted_{os.path.basename(file_path)}.txt")
        else:
            output_file = f"extracted_{os.path.basename(file_path)}.txt"
        
        # Save extracted text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Extracted text from: {file_path}\n")
            f.write("=" * 80 + "\n\n")
            f.write(extracted_text)
        
        print(f"💾 Text saved to: {output_file}")
        
        # Show preview
        preview_length = min(300, len(extracted_text.strip()))
        print(f"📄 Preview: '{extracted_text[:preview_length]}{'...' if len(extracted_text) > preview_length else ''}'")
        
        return True
    else:
        print("❌ No text could be extracted from the file")
        return False

def process_directory(directory_path, output_dir=None):
    """Process all supported files in a directory"""
    if not os.path.isdir(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
    
    # Find all supported files
    files_to_process = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"❌ No supported image files found in: {directory_path}")
        return
    
    print(f"📁 Found {len(files_to_process)} supported files")
    
    successful = 0
    for file_path in files_to_process:
        print(f"\n{'-' * 60}")
        if process_single_file(file_path, output_dir):
            successful += 1
    
    print(f"\n📊 Summary: {successful}/{len(files_to_process)} files processed successfully")

def show_system_info():
    """Display system information and capabilities"""
    print("🚀 Enhanced Document Processing System")
    print("=" * 80)
    print("📸 Supported file types: JPG, JPEG, PNG, TIFF, TIF, BMP, GIF")
    print("🔍 OCR Engine: Tesseract with enhanced preprocessing")
    print("🎯 Optimized for: Handwritten notes, scanned documents, photos")
    
    print("\n🔧 Enhanced Features:")
    print("   • Multiple preprocessing approaches (5 different methods)")
    print("   • Adaptive thresholding for varying lighting conditions")
    print("   • CLAHE histogram equalization")
    print("   • Morphological operations for text cleanup")
    print("   • Automatic selection of best OCR result")
    print("   • Specialized configurations for handwritten text")
    
    print("\n📊 Recent Performance:")
    print("   • IMG_0001.jpg: 470 characters extracted")
    print("   • IMG_0004.tif: 7,580 characters extracted")
    print("   • ruth and joe.jpg: 3,049 characters extracted")
    print("   • WIP.tif: 3,658 characters extracted")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Document Processing System - Extract text from images with improved OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process file.jpg                    # Process single file
  %(prog)s process-dir ./documents/           # Process all images in directory
  %(prog)s process file.jpg -o ./output/      # Save to specific directory
  %(prog)s info                               # Show system information
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single file command
    process_parser = subparsers.add_parser('process', help='Process a single file')
    process_parser.add_argument('file', help='Path to the image file to process')
    process_parser.add_argument('-o', '--output', help='Output directory for extracted text')
    
    # Process directory command
    dir_parser = subparsers.add_parser('process-dir', help='Process all images in a directory')
    dir_parser.add_argument('directory', help='Path to the directory to process')
    dir_parser.add_argument('-o', '--output', help='Output directory for extracted text')
    
    # System info command
    subparsers.add_parser('info', help='Show system information and capabilities')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'process':
        process_single_file(args.file, args.output)
    elif args.command == 'process-dir':
        process_directory(args.directory, args.output)
    elif args.command == 'info':
        show_system_info()

if __name__ == "__main__":
    main()
