"""
Category Analyzer Tool

This script analyzes documents in a corpus to suggest improved categories.
"""

import os
import sys
import re
import random
import logging
import json
from datetime import datetime
from collections import Counter, defaultdict

from llm_analyzer import LLMAnalyzer
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('category_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class CategoryAnalyzer:
    def __init__(self, source_dir, sample_size=1000):
        """Initialize the category analyzer."""
        self.source_dir = source_dir
        self.sample_size = sample_size
        self.llm_analyzer = LLMAnalyzer()
        self.document_processor = DocumentProcessor()
        self.supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.docx', '.doc', '.txt', '.xlsx']
        self.existing_categories = ["Medical Documents", "Receipts", "Contracts", "Photographs", "Other"]
        
        # Use the same LLM instance from the analyzer for category suggestions
        self.llm = self.llm_analyzer.llm
    
    def get_all_files(self):
        """Get all files in the source directory matching supported extensions."""
        all_files = []
        
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    all_files.append(os.path.join(root, file))
        
        logging.info(f"Found {len(all_files)} documents in corpus")
        return all_files
    
    def select_sample(self, all_files):
        """Select a representative sample of files for analysis."""
        if len(all_files) <= self.sample_size:
            return all_files
        
        # Strategy: get a mix of file types and folders
        samples_by_ext = defaultdict(list)
        samples_by_folder = defaultdict(list)
        
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            folder = os.path.basename(os.path.dirname(file_path))
            
            samples_by_ext[ext].append(file_path)
            samples_by_folder[folder].append(file_path)
        
        sample = []
        
        # Get at least one file from each extension type
        for ext, files in samples_by_ext.items():
            sample.append(random.choice(files))
        
        # Get at least one file from each folder
        for folder, files in samples_by_folder.items():
            if not any(f in sample for f in files):
                sample.append(random.choice(files))
        
        # Fill the rest randomly
        remaining_files = [f for f in all_files if f not in sample]
        random.shuffle(remaining_files)
        sample.extend(remaining_files[:self.sample_size - len(sample)])
        
        logging.info(f"Selected {len(sample)} files for analysis")
        return sample
    
    def analyze_file(self, file_path):
        """Analyze a single file using the LLM analyzer."""
        try:
            filename = os.path.basename(file_path)
            creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            
            # Extract text from the file using the document processor
            text = self.document_processor.extract_text(file_path)
            
            result = self.llm_analyzer.analyze_document(text, filename, creation_time, file_path)
            
            # Add the file path and parent folder for context
            result["file_path"] = file_path
            result["parent_folder"] = os.path.basename(os.path.dirname(file_path))
            
            return result
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return None
    
    def analyze_sample(self, sample_files):
        """Analyze a sample of files serially to avoid multiple Ollama requests."""
        results = []
        
        for file in sample_files:
            try:
                result = self.analyze_file(file)
                if result:
                    results.append(result)
                    logging.info(f"Analyzed {file}: {result['category']}")
            except Exception as e:
                logging.error(f"Exception analyzing {file}: {str(e)}")
        
        return results
    
    def suggest_categories(self, analysis_results):
        """Use LLM to suggest improved categories based on analysis results."""
        # Extract current categories and sample document descriptions
        current_categories = [result["category"] for result in analysis_results]
        descriptions = [result["description"] for result in analysis_results]
        parent_folders = [result["parent_folder"] for result in analysis_results]
        
        # Create prompt for LLM
        prompt = f"""
        I need to organize personal and family documents into meaningful categories.
        
        Currently, I'm using these categories:
        {', '.join(self.existing_categories)}
        
        I also have documents organized in these folders:
        {', '.join(set(parent_folders))}
        
        Here are some sample document descriptions from my collection:
        {', '.join(descriptions[:50])}
        
        Some of the current categories assigned to these documents are:
        {', '.join(current_categories[:50])}
        
        Based on this information, please suggest:
        1. An improved list of 8-12 categories that would be most useful for organizing and finding documents
        2. A brief explanation of each category
        3. Examples of what types of documents belong in each category
        
        Focus on categories that are:
        - Mutually exclusive (minimal overlap)
        - Collectively exhaustive (cover all document types)
        - Intuitive for everyday use
        - Useful for quickly finding specific documents
        
        Return your response as a JSON object with this structure:
        {{
            "categories": [
                {{
                    "name": "Category Name",
                    "description": "Brief description",
                    "examples": ["Example doc 1", "Example doc 2"]
                }},
                ...
            ]
        }}
        """
        
        try:
            logging.info("Sending category suggestion prompt to LLM")
            logging.info(f"Prompt: {prompt}")
            # save prompt to a file for debugging
            with open("category_suggestion_prompt.txt", "w") as f:
                f.write(prompt)
            response = self.llm.invoke(prompt)
            logging.info("LLM category suggestion response received")
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```|{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                suggestions = json.loads(json_str)
                return suggestions
            else:
                logging.warning("Could not parse JSON from LLM response")
                return {"categories": [{"name": cat, "description": "", "examples": []} for cat in self.existing_categories]}
        except Exception as e:
            logging.error(f"Error getting category suggestions: {str(e)}")
            return {"categories": [{"name": cat, "description": "", "examples": []} for cat in self.existing_categories]}
    
    def run_analysis(self):
        """Run the complete category analysis process."""
        logging.info(f"Starting category analysis on {self.source_dir}")
        
        # Get all files and select a sample
        all_files = self.get_all_files()
        sample_files = self.select_sample(all_files)
        
        # Analyze the sample
        analysis_results = self.analyze_sample(sample_files)

        # save the analysis_results to a file
        with open("per_file_analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)

        # Get category suggestions
        suggested_categories = self.suggest_categories(analysis_results)
        
        # Save results
        self.save_results(analysis_results, suggested_categories)
        
        return suggested_categories
    
    def save_results(self, analysis_results, suggested_categories):
        """Save analysis results to file."""
        # Compute category distribution
        categories = Counter()
        folder_categories = defaultdict(list)
        
        for result in analysis_results:
            categories[result["category"]] += 1
            folder_categories[result["parent_folder"]].append(result["category"])
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "current_categories": dict(categories),
            "folder_categories": {k: dict(Counter(v)) for k, v in folder_categories.items()},
            "suggested_categories": suggested_categories
        }
        
        with open("category_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logging.info("Analysis results saved to category_analysis_results.json")


if __name__ == "__main__":
    # Default source directory
    source_dir = r"E:\Dropbox\Admin\Scanned Documents"
    
    # Allow overriding from command line
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    
    # Create and run the analyzer
    sample_size = 10
    sample_size = sys.argv[2]   if len(sys.argv) > 2 else sample_size
    try:
        sample_size = int(sample_size)
    except ValueError:
        logging.error(f"Invalid sample size '{sample_size}', using default {10}")
        sample_size = 10
    analyzer = CategoryAnalyzer(source_dir, sample_size)
    suggested_categories = analyzer.run_analysis()
    
    print("\nSuggested Document Categories:")
    print("=============================")
    
    if "categories" in suggested_categories:
        for cat in suggested_categories["categories"]:
            print(f"\n{cat['name']}")
            print(f"  Description: {cat['description']}")
            print(f"  Examples: {', '.join(cat['examples'])}")
    else:
        print("\nNo categories were suggested. Check the logs for details.")
    
    print("\nComplete analysis results saved to category_analysis_results.json")
