import os
import sys
import random
import logging
import json
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_analyzer import LLMAnalyzer
from langchain_ollama import OllamaLLM

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
    def __init__(self, source_dir, sample_size=100, max_workers=4):
        """
        Initialize the category analyzer.
        
        Args:
            source_dir (str): Path to the document corpus
            sample_size (int): Number of documents to sample for analysis
            max_workers (int): Maximum number of concurrent workers for analysis
        """
        self.source_dir = source_dir
        self.sample_size = sample_size
        self.max_workers = max_workers
        self.llm_analyzer = LLMAnalyzer()
        self.supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.docx', '.doc', '.txt']
        self.existing_categories = ["Medical Documents", "Receipts", "Contracts", "Photographs", "Other"]
        
        # For direct LLM calls without using the analyzer
        try:
            # self.llm = OllamaLLM(model="gemma3:27b-it-fp16")
            self.llm = OllamaLLM(model="qwq:32b-fp16")
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            self.llm = None
    
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
        
        # Ensure we get a mix of file types and folders
        samples_by_ext = defaultdict(list)
        samples_by_folder = defaultdict(list)
        
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            folder = os.path.basename(os.path.dirname(file_path))
            
            samples_by_ext[ext].append(file_path)
            samples_by_folder[folder].append(file_path)
        
        # Strategy: take a stratified sample across extensions and folders
        sample = []
        
        # First, ensure we have at least one file from each extension type
        for ext, files in samples_by_ext.items():
            sample.append(random.choice(files))
        
        # Then, ensure we have at least one file from each folder
        for folder, files in samples_by_folder.items():
            if not any(f in sample for f in files):
                sample.append(random.choice(files))
        
        # Fill the rest randomly until we reach the sample size
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
            
            # For simplicity in this script, we'll skip text extraction
            # and rely on image analysis for all files
            text = ""  # We'll rely on visual analysis for images
            
            result = self.llm_analyzer.analyze_document(text, filename, creation_time, file_path)
            
            # Add the file path and parent folder for context
            result["file_path"] = file_path
            result["parent_folder"] = os.path.basename(os.path.dirname(file_path))
            
            return result
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return None
    
    def analyze_sample(self, sample_files):
        """Analyze a sample of files concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.analyze_file, file): file for file in sample_files}
            
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logging.info(f"Analyzed {file}: {result['category']}")
                except Exception as e:
                    logging.error(f"Exception analyzing {file}: {str(e)}")
        
        return results
    
    def extract_existing_categories(self, analysis_results):
        """Extract categories from analysis results."""
        categories = Counter()
        folder_categories = defaultdict(list)
        
        for result in analysis_results:
            categories[result["category"]] += 1
            folder_categories[result["parent_folder"]].append(result["category"])
        
        logging.info(f"Current category distribution: {categories}")
        logging.info(f"Folder to category mapping: {dict(folder_categories)}")
        
        return categories, folder_categories
    
    def suggest_categories(self, analysis_results):
        """Use LLM to suggest improved categories based on analysis results."""
        if not self.llm:
            logging.error("LLM not initialized, cannot suggest categories")
            return self.existing_categories
        
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
            response = self.llm.invoke(prompt)
            logging.info("LLM category suggestion response received")
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```|{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                suggestions = json.loads(json_str)
                
                logging.info(f"Suggested categories: {[cat['name'] for cat in suggestions['categories']]}")
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
        
        # Extract existing categories
        categories, folder_categories = self.extract_existing_categories(analysis_results)
        
        # Get category suggestions
        suggested_categories = self.suggest_categories(analysis_results)
        
        # Save results
        self.save_results(categories, folder_categories, suggested_categories)
        
        return suggested_categories
    
    def save_results(self, categories, folder_categories, suggested_categories):
        """Save analysis results to file."""
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
    import re  # Import re at the top level for JSON parsing
    
    # Default source directory
    source_dir = r"E:\Dropbox\Admin\Scanned Documents"
    
    # Allow overriding from command line
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    
    # Adjust sample size based on expected corpus size
    sample_size = 100
    
    analyzer = CategoryAnalyzer(source_dir, sample_size)
    suggested_categories = analyzer.run_analysis()
    
    print("\nSuggested Document Categories:")
    print("=============================")
    
    for cat in suggested_categories["categories"]:
        print(f"\n{cat['name']}")
        print(f"  Description: {cat['description']}")
        print(f"  Examples: {', '.join(cat['examples'])}")
    
    print("\nComplete analysis results saved to category_analysis_results.json")
