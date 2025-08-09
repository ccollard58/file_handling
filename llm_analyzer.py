import re
import os
import logging
import json
import requests
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMAnalyzer:
    def __init__(self, model="gemma3:latest", temperature=0.6, vision_model="llava:latest"):
        self.model = model
        self.temperature = temperature
        self.vision_model = vision_model
        self.llm = None
        self.vision_llm = None
        self.initialize_llm()
        self.initialize_vision_llm()

    def initialize_llm(self):
        """Initializes the OllamaLLM instance."""
        try:
            self.llm = OllamaLLM(model=self.model, temperature=self.temperature)
            logging.info(f"LLM initialized successfully with model {self.model} and temperature {self.temperature}")
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            self.llm = None  # Ensure llm is None if initialization fails
            raise

    def initialize_vision_llm(self):
        """Initializes the vision-capable OllamaLLM instance with temperature 0.0 for deterministic analysis."""
        try:
            self.vision_llm = OllamaLLM(model=self.vision_model, temperature=0.0)
            logging.info(f"Vision LLM initialized successfully with model {self.vision_model} and temperature 0.0")
        except Exception as e:
            logging.error(f"Error initializing Vision LLM: {str(e)}")
            self.vision_llm = None

    def update_settings(self, model, temperature, vision_model=None):
        """Updates the LLM model and temperature and re-initializes the LLM."""
        self.model = model
        self.temperature = float(temperature)
        if vision_model:
            self.vision_model = vision_model
        self.initialize_llm()
        self.initialize_vision_llm()

    @staticmethod
    def get_available_models(ollama_base_url="http://localhost:11434"):
        """Fetches available models from the Ollama API."""
        try:
            response = requests.get(f"{ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Ollama models: {e}")
            return []

    @staticmethod
    def get_text_models(ollama_base_url="http://localhost:11434"):
        """Fetches text/reasoning models from the Ollama API (excludes vision models)."""
        try:
            response = requests.get(f"{ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            all_models = [model["name"] for model in models]
            
            # Filter out vision-capable models (common vision model patterns)
            vision_keywords = ['gemma3', 'qwen2.5vl', 'mistral-small3.1', 'llava', 'vision', 'minicpm', 'moondream', 'bakllava', 'cogvlm']
            text_models = []
            
            for model in all_models:
                model_lower = model.lower()
                # Include model only if it doesn't contain vision keywords
                if not any(keyword in model_lower for keyword in vision_keywords):
                    text_models.append(model)
            
            # If no text models found, return all models as fallback
            return text_models if text_models else all_models
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Ollama text models: {e}")
            return []

    @staticmethod
    def get_vision_models(ollama_base_url="http://localhost:11434"):
        """Fetches vision-capable models from the Ollama API."""
        try:
            response = requests.get(f"{ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            all_models = [model["name"] for model in models]
            
            # Filter for vision-capable models (common vision model patterns)
            vision_keywords = ['gemma3', 'qwen2.5vl', 'mistral-small3.1', 'llava', 'vision', 'minicpm', 'moondream', 'bakllava', 'cogvlm']
            vision_models = []
            
            for model in all_models:
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in vision_keywords):
                    vision_models.append(model)
            
            # If no vision models found, return all models as fallback
            return vision_models if vision_models else all_models
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Ollama vision models: {e}")
            return []
    
    def analyze_document(self, text, filename, creation_date, file_path=None):
        """
        Analyze document text to extract key information.
        
        Returns:
            dict: Contains identity, date, description, and category
        """
        if not self.llm:
            logging.error("LLM not initialized, cannot analyze document.")
            return self._create_default_result(filename, creation_date)

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
        """Analyze an image file visually using Vision LLM."""
        if not self.vision_llm:
            logging.error("Vision LLM not initialized, cannot analyze image.")
            return self._create_image_fallback_result(filename, creation_date)
        try:
            # Create prompt for visual analysis
            prompt = """Analyze this image to extract document information. Look for:
            1. Any visible text or names (especially "Chuck Collard", "Charles Collard", "Charles W Collard", "Colleen McGinnis", or "Colleen Collard")
            2. What type of document this appears to be
            3. Any visible dates
            4. A brief descriptive title for the document (5 words or less)

            For the category, suggest the BEST category name that describes what type of document this is. Use one of the following categories with their descriptions and examples:
            - "Medical": Documents related to personal and family health, including prescriptions, exam results, insurance information, and wellness records. Examples: Medical Imaging Reports, Lab Results, Physical Therapy Plans, Dulera Prescription, Eye Exam Prescription, Pupil Distance Waiver Form
            - "Identification": Passports, driver's licenses, IDs, and vital records. Examples: Passport, Driver's License, Birth Certificate, Social Security Card
            - "Home": Documents related to your residence, including purchase agreements, maintenance records, utilities and property information. Examples: Home Warranty, Property Tax Documents, Construction Permits, Mortgage Papers, Closing Documents, Homeowner Insurance, Electricity Bills, Cable Bills
            - "Auto": Car titles, maintenance records, and vehicle-related paperwork. Examples: Car Title, Auto Repair Records, Registration Documents, Insurance Claim Forms, BMW Warranty Extension Details
            - "SysAdmin": Documents related to software, network configurations, and technical instructions, including Software licenses, user manuals, and tech warranties. Examples: Software Licenses, Hardware Specifications, Appliance Manuals, Product Warranties, Network Configuration Diagram, Technical Error Report
            - "School": Degrees, transcripts, and academic records. Examples: Degree Certificates, Transcripts, Course Materials, Student Loans Documents, FranklinCovey Training Notes
            - "Cooking": Collection of recipes, cookbooks, meal plans and related culinary information. Examples: Apple Raisin Crisp Recipe, Cooking Recipes, Meal Plans, Diet Guides
            - "Financial": Documents related to income, expenses, investments, and taxes. Examples: W-2s, Wills, Tax Documents, Bank Statements, Pay Stubs, Investment Records
            - "Travel": Documents related to trips, vacations, and recreational activities. Examples: Travel Itineraries, Boarding Passes, Hotel Confirmation, Tourism Information, Trip Insurance
            - "Employment": Documents related to employment history, benefits, and income. Examples: Pay stubs, Employment Contracts, Benefits Forms, Performance Reviews
            - "Hobbies": Documents related to personal hobbies and interests. Examples: DIY Guides, Craft Patterns, Art Supply Inventories, Media Releases, Copyright Registrations
            - "Memories": Documents that capture a fond memory. Examples: Letters, Notes, Theater Ticket Stubs, Photographs
            - "Other": Documents that don't fit neatly into other categories, or are unclear in purpose. Examples: Chinese Text Document, Abstract colorful image, Franklin Institute Color Codes, Roadside Attraction Sign

            Based on what you can see in this image, respond in JSON format:
             {
                 "identity": "your guess here (Chuck or Colleen or Unknown)",
                 "description": "Brief descriptive title",
                 "category": "Your best category suggestion here",
                 "visible_text": "Any text you can clearly read",
                 "document_type": "What type of document this appears to be"
             }"""
            
            # Use the file path directly with the images parameter (like in the notebook)
            response = self.vision_llm.invoke(prompt, images=[str(file_path)])
            logging.debug(f"Vision LLM response for image {filename}: {response}")
        
            # Try to extract JSON from response
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Extract date from filename or use creation date
                date = self._extract_date("", filename, creation_date)
                # If image has no visible text, treat as photo
                visible = result.get("visible_text", "").strip()
                if not visible:
                    # Clean filename for description
                    base = os.path.splitext(filename)[0]
                    desc = base.replace('_', ' ').replace('-', ' ').title()
                    return {
                        "identity": "Unknown",
                        "date": date,
                        "description": desc,
                        "category": "Photography"
                    }
                # Otherwise use LLM result
                return {
                    "identity": result.get("identity", "Unknown"),
                    "date": date,
                    "description": result.get("description", "Image Document"),
                    "category": result.get("category", "Uncategorized")
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
        # Try to infer category from filename based on user-defined categories
        filename_lower = filename.lower()
        
        # Define category keywords with updated definitions
        if any(word in filename_lower for word in ['medical', 'prescription', 'lab', 'clinic', 'health', 'imaging', 'wellness']):
            category = "Medical"
        elif any(word in filename_lower for word in ['passport', 'driver', 'license', 'id', 'birth', 'social', 'vital']):
            category = "Identification"
        elif any(word in filename_lower for word in ['home', 'warranty', 'property', 'tax', 'permit', 'mortgage', 'closing', 'insurance', 'electricity', 'cable', 'utilities', 'maintenance']):
            category = "Home"
        elif any(word in filename_lower for word in ['car', 'auto', 'vehicle', 'registration', 'repair', 'title', 'insurance', 'claim', 'maintenance']):
            category = "Auto"
        elif any(word in filename_lower for word in ['software', 'license', 'manual', 'warranty', 'network', 'config', 'technical', 'appliance', 'hardware', 'diagram', 'error']):
            category = "SysAdmin"
        elif any(word in filename_lower for word in ['degree', 'transcript', 'course', 'student', 'training', 'academic']):
            category = "School"
        elif any(word in filename_lower for word in ['recipe', 'cookbook', 'meal', 'plan', 'diet', 'culinary']):
            category = "Cooking"
        elif any(word in filename_lower for word in ['bank', 'statement', 'invoice', 'tax', 'will', 'paystub', 'payment', 'investment']):
            category = "Financial"
        elif any(word in filename_lower for word in ['itinerary', 'ticket', 'boarding', 'hotel', 'trip', 'tourism', 'reservation', 'vacation']):
            category = "Travel"
        elif any(word in filename_lower for word in ['employment', 'contract', 'pay', 'benefit', 'review', 'history', 'performance']):
            category = "Employment"
        elif any(word in filename_lower for word in ['diy', 'guide', 'craft', 'pattern', 'art', 'hobby', 'media', 'copyright']):
            category = "Hobbies"
        elif any(word in filename_lower for word in ['letter', 'note', 'ticket', 'stub', 'memory', 'photograph', 'photographs', 'photo']):
            category = "Memories"
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
        
        if not self.llm:
            logging.warning("LLM not initialized, returning 'Unknown' for identity.")
            return "Unknown"

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
        if not self.llm:
            logging.error("LLM not initialized, cannot analyze document content.")
            return {"description": "Unknown Document", "category": "Uncategorized"}

        prompt = PromptTemplate(
            input_variables=["text", "filename"],
            template="""
            Analyze the following document text and filename to:
            1. Create a brief descriptive title (5 words or less)
            2. Suggest the BEST category name for this document using one of the defined categories below:
            
            - "Medical": Documents related to personal and family health, including prescriptions, exam results, insurance information, and wellness records. Examples: Medical Imaging Reports, Lab Results, Physical Therapy Plans, Dulera Prescription, Eye Exam Prescription, Pupil Distance Waiver Form
            - "Identification": Passports, driver's licenses, IDs, and vital records. Examples: Passport, Driver's License, Birth Certificate, Social Security Card
            - "Home": Documents related to your residence, including purchase agreements, maintenance records, utilities and property information. Examples: Home Warranty, Property Tax Documents, Construction Permits, Mortgage Papers, Closing Documents, Homeowner Insurance, Electricity Bills, Cable Bills
            - "Auto": Car titles, maintenance records, and vehicle-related paperwork. Examples: Car Title, Auto Repair Records, Registration Documents, Insurance Claim Forms, BMW Warranty Extension Details
            - "SysAdmin": Documents related to software, network configurations, and technical instructions, including Software licenses, user manuals, and tech warranties. Examples: Software Licenses, Hardware Specifications, Appliance Manuals, Product Warranties, Network Configuration Diagram, Technical Error Report
            - "School": Degrees, transcripts, and academic records. Examples: Degree Certificates, Transcripts, Course Materials, Student Loans Documents, FranklinCovey Training Notes
            - "Cooking": Collection of recipes, cookbooks, meal plans and related culinary information. Examples: Apple Raisin Crisp Recipe, Cooking Recipes, Meal Plans, Diet Guides
            - "Financial": Documents related to income, expenses, investments, and taxes. Examples: W-2s, Wills, Tax Documents, Bank Statements, Pay Stubs, Investment Records
            - "Travel": Documents related to trips, vacations, and recreational activities. Examples: Travel Itineraries, Boarding Passes, Hotel Confirmation, Tourism Information, Trip Insurance
            - "Employment": Documents related to employment history, benefits, and income. Examples: Pay Stubs, Employment Contracts, Benefits Forms, Performance Reviews
            - "Hobbies": Documents related to personal hobbies and interests. Examples: DIY Guides, Craft Patterns, Art Supply Inventories, Media Releases, Copyright Registrations
            - "Memories": Documents that capture a fond memory. Examples: Letters, Notes, Theater Ticket Stubs, Photographs
            - "Other": Documents that don't fit neatly into other categories, or are unclear in purpose. Examples: Chinese Text Document, Abstract colorful image, Franklin Institute Color Codes, Roadside Attraction Sign
            
            Document filename: {filename}
            Document text (partial):
            {text}
            
            Respond in JSON format:
            {{"description": "Brief title here", "category": "One of the categories above"}}
            """
        )
        
        try:
            response = self.llm.invoke(prompt.format(text=text[:2000], filename=filename))
            
            # Try to extract JSON from response
            logging.debug(f"LLM response for document {filename}: {response}")
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "description": result.get("description", "Unknown Document"),
                    "category": result.get("category", "Uncategorized")
                }
            else:
                # Fallback parsing if JSON extraction fails
                description_match = re.search(r'"description":\s*"([^"]+)"', response)
                category_match = re.search(r'"category":\s*"([^"]+)"', response)
                
                return {
                    "description": description_match.group(1) if description_match else "Unknown Document",
                    "category": category_match.group(1) if category_match else "Uncategorized"
                }
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return {"description": "Unknown Document", "category": "Uncategorized"}
    
    def _create_default_result(self, filename, creation_date):
        """Create a default result when analysis fails."""
        return {
            "identity": "Unknown",
            "date": creation_date.strftime('%Y-%m-%d'),
            "description": os.path.splitext(os.path.basename(filename))[0],
            "category": "Uncategorized"
        }
