import re
import os
import logging
import json
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMAnalyzer:
    def __init__(self):
        try:
            # Use the same model from the notebook that works with images
            self.llm = OllamaLLM(model="gemma3:27b-it-fp16")
            logging.info("LLM initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def analyze_document(self, text, filename, creation_date, file_path=None):
        """
        Analyze document text to extract key information.
        
        Returns:
            dict: Contains identity, date, description, and category
        """
        # Check if this is an image file
        is_image_file = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'))
        
        # Check if OCR text is insufficient or likely just noise
        has_insufficient_text = not text or len(text.strip()) < 10
        
        # Additional check for .tif files and other images that might return noisy OCR
        is_likely_noise = False
        if is_image_file and text and len(text.strip()) < 50:
            # Count ratio of alphanumeric characters vs. total non-whitespace characters
            non_whitespace = len(text.strip())
            alphanumeric = sum(c.isalnum() for c in text)
            # If less than 40% of characters are alphanumeric, likely noise
            is_likely_noise = (alphanumeric / non_whitespace) < 0.4 if non_whitespace > 0 else True
            
            # Specific handling for .tif files which often need visual analysis
            if filename.lower().endswith(('.tif', '.tiff')) and len(text.strip()) < 30:
                is_likely_noise = True
                logging.info(f".tif/.tiff file {filename} detected with minimal text, preferring visual analysis")
        
        if (has_insufficient_text or is_likely_noise) and is_image_file and file_path:
            logging.info(f"Insufficient or noisy OCR text for image {filename}, attempting visual analysis")
            try:
                return self._analyze_image_visually(file_path, filename, creation_date)
            except Exception as e:
                logging.error(f"Error in visual image analysis for {filename}: {str(e)}")
                return self._create_default_result(filename, creation_date)
        elif has_insufficient_text:
            logging.warning(f"Insufficient text to analyze for {filename}")
            return self._create_default_result(filename, creation_date)
        
        try:
            # First identify the person (Chuck or Colleen)
            identity = self._identify_person(text)
            
            # Then extract the document date
            date = self._extract_date(text, filename, creation_date)
            
            # Analyze document for category and description
            doc_info = self._analyze_document_content(text, filename)
            
            return {
                "identity": identity,
                "date": date,
                "description": doc_info["description"],
                "category": doc_info["category"]
            }
        except Exception as e:
            logging.error(f"Error analyzing document {filename}: {str(e)}")
            return self._create_default_result(filename, creation_date)
    def _analyze_image_visually(self, file_path, filename, creation_date):
        """Analyze an image file visually using LLM."""
        try:
            # Create prompt for visual analysis
            prompt = """Analyze this image to extract document information. Look for:
1. Any visible text or names (especially "Chuck Collard", "Charles Collard", "Charles W Collard", "Colleen McGinnis", or "Colleen Collard")
2. What type of document this appears to be (medical document, receipt, contract, photograph, or other)
3. Any visible dates
4. A brief descriptive title for the document (5 words or less)

Important: If this appears to be a personal photograph (family photos, vacation pictures, portraits, etc.) rather than a document with text, categorize it as "Photographs".

Based on what you can see in this image, respond in JSON format:
{
    "identity": "Chuck or Colleen or Unknown",
    "description": "Brief descriptive title",
    "category": "Medical Documents or Receipts or Contracts or Photographs or Other",
    "visible_text": "Any text you can clearly read",
    "document_type": "What type of document this appears to be"
}"""
            
            # Use the file path directly with the images parameter (like in the notebook)
            response = self.llm.invoke(prompt, images=[str(file_path)])
            logging.info(f"LLM response for image {filename}: {response}")
            
            # Try to extract JSON from response
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Extract date from filename or use creation date
                date = self._extract_date("", filename, creation_date)
                
                return {
                    "identity": result.get("identity", "Unknown"),
                    "date": date,
                    "description": result.get("description", "Image Document"),
                    "category": result.get("category", "Other")
                }
            else:
                # Fallback if JSON parsing fails
                logging.warning(f"Could not parse LLM response for image {filename}")
                return self._create_image_fallback_result(filename, creation_date)
                
        except Exception as e:
            logging.error(f"Error in visual image analysis: {str(e)}")
            return self._create_image_fallback_result(filename, creation_date)
    def _create_image_fallback_result(self, filename, creation_date):
        """Create a fallback result for images when visual analysis fails."""
        # Try to infer category from filename
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['medical', 'doctor', 'prescription', 'hospital', 'clinic']):
            category = "Medical Documents"
        elif any(word in filename_lower for word in ['receipt', 'invoice', 'bill', 'purchase']):
            category = "Receipts"
        elif any(word in filename_lower for word in ['contract', 'agreement', 'legal']):
            category = "Contracts"
        elif any(word in filename_lower for word in ['photo', 'pic', 'img', 'image', 'portrait', 'family', 'vacation', 'selfie', 'snapshot']):
            category = "Photographs"
        else:
            # For image files, default to Photographs unless there are clear document indicators
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']:
                category = "Photographs"
            else:
                category = "Other"
        
        # Extract base filename without extension for description
        base_name = os.path.splitext(filename)[0]
        # Clean up the filename for a better description
        description = base_name.replace('_', ' ').replace('-', ' ').title()
        if len(description) > 30:
            description = description[:30] + "..."
        
        return {
            "identity": "Unknown",
            "date": creation_date.strftime('%Y-%m-%d'),
            "description": description,
            "category": category
        }

    def _identify_person(self, text):
        """Identify if the document belongs to Chuck or Colleen."""
        chuck_patterns = [
            r"Charles\s+Collard", r"Chuck\s+Collard", r"Charles\s+W\.?\s+Collard"
        ]
        
        colleen_patterns = [
            r"Colleen\s+McGinnis", r"Colleen\s+Collard"
        ]
        
        # Check for Chuck
        for pattern in chuck_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Chuck"
        
        # Check for Colleen
        for pattern in colleen_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Colleen"
        
        # If no match, use LLM to determine the most likely person
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Based on the following document text, determine if it most likely belongs to "Chuck Collard" 
            (also known as "Charles Collard" or "Charles W Collard") or "Colleen McGinnis" 
            (also known as "Colleen Collard").
            
            Document text:
            {text}
            
            Answer with only "Chuck" or "Colleen" or "Unknown":
            """
        )
        
        response = self.llm.invoke(prompt.format(text=text[:1000]))  # Use first 1000 chars for efficiency
        
        if "chuck" in response.lower():
            return "Chuck"
        elif "colleen" in response.lower():
            return "Colleen"
        else:
            return "Unknown"
    
    def _extract_date(self, text, filename, creation_date):
        """
        Extract the document date in order of priority:
        1. Transaction/generation date in the document
        2. Date in the filename
        3. File creation date
        """
        # Try to find date in document text
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',    # YYYY/MM/DD
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})',  # Month DD, YYYY
            r'(\d{1,2}) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{4})'  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Process the first match (this is simplistic and might need refinement)
                try:
                    # This is a simplified approach - would need to be expanded based on which pattern matched
                    date_str = '/'.join([str(x) for x in matches[0] if x])
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    pass
        
        # Try to find date in filename
        filename_date_pattern = r'(\d{4})[-_]?(\d{1,2})[-_]?(\d{1,2})'
        filename_matches = re.search(filename_date_pattern, filename)
        if filename_matches:
            try:
                year, month, day = filename_matches.groups()
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%Y-%m-%d')
            except:
                pass
        
        # Use file creation date as fallback
        return creation_date.strftime('%Y-%m-%d')
    
    def _analyze_document_content(self, text, filename):
        """Use LLM to analyze document content and determine category and description."""
        prompt = PromptTemplate(
            input_variables=["text", "filename"],
            template="""
            Analyze the following document text and filename to:
            1. Create a brief descriptive title (5 words or less)
            2. Determine the category from these options: "Medical Documents", "Receipts", "Contracts", "Other"
            
            Document filename: {filename}
            Document text (partial):
            {text}
            
            Respond in JSON format:
            {{"description": "Brief title here", "category": "Category here"}}
            """
        )
        
        try:
            response = self.llm.invoke(prompt.format(text=text[:2000], filename=filename))
            
            # Try to extract JSON from response
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "description": result.get("description", "Unknown Document"),
                    "category": result.get("category", "Other")
                }
            else:
                # Fallback parsing if JSON extraction fails
                description_match = re.search(r'"description":\s*"([^"]+)"', response)
                category_match = re.search(r'"category":\s*"([^"]+)"', response)
                
                return {
                    "description": description_match.group(1) if description_match else "Unknown Document",
                    "category": category_match.group(1) if category_match else "Other"
                }
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return {"description": "Unknown Document", "category": "Other"}
    
    def _create_default_result(self, filename, creation_date):
        """Create a default result when analysis fails."""
        return {
            "identity": "Unknown",
            "date": creation_date.strftime('%Y-%m-%d'),
            "description": os.path.splitext(os.path.basename(filename))[0],
            "category": "Other"
        }
