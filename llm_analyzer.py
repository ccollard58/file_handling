import re
import os
import logging
import json
import requests
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from PIL import Image
import tempfile
import base64
import mimetypes
from typing import Optional

class LLMAnalyzer:
    def __init__(self, model="gemma3:latest", temperature=0.6, vision_model="llava:latest"):
        self.model = model
        self.temperature = temperature
        self.vision_model = vision_model
        self.llm = None
        self.vision_llm = None
        self.initialize_llm()
        self.initialize_vision_llm()

    @staticmethod
    def _as_text(response):
        """Best-effort extraction of text from a LangChain response or raw string."""
        try:
            if hasattr(response, "text") and callable(getattr(response, "text")):
                return response.text()
            if hasattr(response, "content"):
                return response.content if isinstance(response.content, str) else str(response.content)
            return str(response)
        except Exception:
            return str(response)

    @staticmethod
    def _make_file_url(path_str: str) -> str:
        """Convert a local file path to a file:/// URL suitable for vision models."""
        abs_path = os.path.abspath(path_str)
        return "file:///" + abs_path.replace("\\", "/")

    def _prepare_image_path(self, file_path: str) -> str:
        """If a TIFF image is provided, convert to PNG for better compatibility. Returns a path to use."""
        lower = str(file_path).lower()
        if lower.endswith((".tif", ".tiff")):
            try:
                with Image.open(file_path) as im:
                    # Convert to RGB to ensure compatibility
                    if im.mode not in ("RGB", "RGBA"):
                        im = im.convert("RGB")
                    # Save to a temp PNG file
                    tmp_dir = tempfile.gettempdir()
                    base = os.path.splitext(os.path.basename(file_path))[0]
                    out_path = os.path.join(tmp_dir, f"{base}_converted.png")
                    im.save(out_path, format="PNG")
                    logging.info(f"Converted TIFF to PNG for vision analysis: {out_path}")
                    return out_path
            except Exception as conv_err:
                logging.warning(f"Failed to convert TIFF to PNG, will try original file: {conv_err}")
                return str(file_path)
        return str(file_path)

    @staticmethod
    def _image_to_data_url(path_str: str) -> str:
        """Read an image file and return a base64 data URL string."""
        mime, _ = mimetypes.guess_type(path_str)
        # Default to PNG if unknown
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        with open(path_str, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _image_to_base64(path_str: str) -> str:
        """Return the raw base64-encoded contents of an image (no data: prefix)."""
        with open(path_str, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _is_qwen_model(self, model_name):
        """Check if the model is a Qwen model that requires specific hyperparameters."""
        if not model_name:
            return False
        
        model_lower = model_name.lower()
        qwen_patterns = [
            'qwen',
            'qwq'
        ]
        
        return any(pattern in model_lower for pattern in qwen_patterns)

    def initialize_llm(self):
        """Initializes the ChatOllama instance."""
        try:
            # Check if this is a Qwen model and use specific hyperparameters
            if self._is_qwen_model(self.model):
                self.llm = ChatOllama(
                    model=self.model, 
                    temperature=self.temperature,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0
                )
                logging.info(f"LLM initialized successfully with Qwen model {self.model}, temperature {self.temperature}, top_p=0.8, top_k=20, min_p=0.0")
            else:
                self.llm = ChatOllama(model=self.model, temperature=self.temperature)
                logging.info(f"LLM initialized successfully with model {self.model} and temperature {self.temperature}")
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            self.llm = None  # Ensure llm is None if initialization fails
            raise

    def initialize_vision_llm(self):
        """Initializes the vision-capable ChatOllama instance with temperature 0.0 for deterministic analysis."""
        try:
            # Check if this is a Qwen vision model and use specific hyperparameters
            if self._is_qwen_model(self.vision_model):
                self.vision_llm = ChatOllama(
                    model=self.vision_model, 
                    temperature=0.0,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0
                )
                logging.info(f"Vision LLM initialized successfully with Qwen model {self.vision_model}, temperature 0.0, top_p=0.8, top_k=20, min_p=0.0")
            else:
                self.vision_llm = ChatOllama(model=self.vision_model, temperature=0.0)
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
            
            # Filter out vision-capable models using updated list from Ollama
            vision_keywords = [
                'llava',           # LLaVA models (llava, llava-llama3, llava-phi3)
                'vision',          # Models with 'vision' in name (llama3.2-vision, granite3.2-vision)
                'minicpm-v',       # MiniCPM vision models
                'qwen2.5vl',       # Qwen vision-language models
                'moondream',       # Moondream vision models
                'bakllava',        # BakLLaVA models
                'mistral-small3',  # Mistral Small 3.1/3.2 with vision
                'granite3.2-vision', # Granite vision models
                'cogvlm',          # CogVLM models
                'pixtral',         # Pixtral models
                'gemma3',          # Gemma3 has vision capabilities (some variants are dual-use)
                'llama4'          # Llama4 multimodal models (some variants are dual-use)
            ]
            # Some models can be used both as text and vision; include them in BOTH lists
            dual_keywords = [
                'gemma3n',  # gemma3n variants can be text+vision
                'gemma3',   # gemma3 family (treat as dual-use by default)
            ]
            text_models = []
            
            for model in all_models:
                model_lower = model.lower()
                is_vision = any(keyword in model_lower for keyword in vision_keywords)
                is_dual = any(keyword in model_lower for keyword in dual_keywords)
                # Include text models that aren't vision-only, and always include dual-use models
                if (not is_vision) or is_dual:
                    text_models.append(model)
            
            # If no text models found, return all models as fallback
            # De-duplicate while preserving order
            seen = set()
            deduped = [m for m in text_models if not (m in seen or seen.add(m))]
            return deduped if deduped else all_models
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
            
            # Filter for vision-capable models based on Ollama's vision models
            # Updated list from https://ollama.com/search?c=vision
            vision_keywords = [
                'llava',           # LLaVA models (llava, llava-llama3, llava-phi3)
                'vision',          # Models with 'vision' in name (llama3.2-vision, granite3.2-vision)
                'minicpm-v',       # MiniCPM vision models
                'qwen2.5vl',       # Qwen vision-language models
                'moondream',       # Moondream vision models
                'bakllava',        # BakLLaVA models
                'mistral-small3',  # Mistral Small 3.1/3.2 with vision
                'granite3.2-vision', # Granite vision models
                'cogvlm',          # CogVLM models
                'pixtral',         # Pixtral models
                'gemma3',          # Gemma3 has vision capabilities (some variants are dual-use)
                'llama4'           # Llama4 multimodal models (some variants are dual-use)
            ]
            # Models that are known to be dual-use (text + vision) — include them here as well
            dual_keywords = [
                'gemma3n',
                'gemma3',
            ]
            vision_models = []
            
            for model in all_models:
                model_lower = model.lower()
                is_vision = any(keyword in model_lower for keyword in vision_keywords)
                is_dual = any(keyword in model_lower for keyword in dual_keywords)
                if is_vision or is_dual:
                    vision_models.append(model)
            
            # If no vision models found, return all models as fallback
            # De-duplicate while preserving order
            seen = set()
            deduped = [m for m in vision_models if not (m in seen or seen.add(m))]
            return deduped if deduped else all_models
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
            prompt = """Analyze this image to extract document information. Focus on:
            1. Extract ALL visible text accurately, including any names, addresses, dates, amounts, or other important information
            2. What type of document this appears to be
            3. Any visible dates
            4. A brief descriptive title for the document (5 words or less)

            SPECIAL INSTRUCTIONS FOR PAYSTUBS:
            - If this is a paystub/pay stub/earnings statement, use EXACTLY "Pay Stub" as the description (add stub number if visible, e.g., "Pay Stub A12345")
            - Paystubs should ALWAYS be categorized as "Employment" (NOT "Financial")
            - Look for paystub indicators: gross pay, net pay, deductions, federal tax, pay period, hours worked, etc.

            For the category, suggest the BEST category name that describes what type of document this is. Use one of the following categories with their descriptions and examples:
            - "Medical": Documents related to personal and family health, including prescriptions, exam results, insurance information, and wellness records. Examples: Medical Imaging Reports, Lab Results, Physical Therapy Plans, Dulera Prescription, Eye Exam Prescription, Pupil Distance Waiver Form
            - "Identification": Passports, driver's licenses, IDs, and vital records. Examples: Passport, Driver's License, Birth Certificate, Social Security Card
            - "Home": Documents related to your residence, including purchase agreements, maintenance records, utilities, property information, and plant/gardening activities. Examples: Home Warranty, Property Tax Documents, Construction Permits, Mortgage Papers, Closing Documents, Homeowner Insurance, Electricity Bills, Cable Bills, Plant Care Guides, Garden Plans, Landscaping Documents
            - "Auto": Car titles, maintenance records, and vehicle-related paperwork. Examples: Car Title, Auto Repair Records, Registration Documents, Insurance Claim Forms, BMW Warranty Extension Details
            - "SysAdmin": Documents related to software, network configurations, and technical instructions, including Software licenses, user manuals, and tech warranties. Examples: Software Licenses, Hardware Specifications, Appliance Manuals, Product Warranties, Network Configuration Diagram, Technical Error Report
            - "School": Degrees, transcripts, and academic records. Examples: Degree Certificates, Transcripts, Course Materials, Student Loans Documents, FranklinCovey Training Notes
            - "Cooking": Collection of recipes, cookbooks, meal plans and related culinary information. Examples: Apple Raisin Crisp Recipe, Cooking Recipes, Meal Plans, Diet Guides
            - "Financial": Documents related to income, expenses, investments, and taxes. Examples: W-2s, Wills, Tax Documents, Bank Statements, Investment Records (NOTE: Pay stubs go in Employment, not Financial)
            - "Travel": Documents related to trips, vacations, and recreational activities. Examples: Travel Itineraries, Boarding Passes, Hotel Confirmation, Tourism Information, Trip Insurance
            - "Employment": Documents related to employment history, benefits, and income. Examples: Pay Stubs, Employment Contracts, Benefits Forms, Performance Reviews (NOTE: This is the correct category for paystubs)
            - "Hobbies": Documents related to personal hobbies and interests. Examples: DIY Guides, Craft Patterns, Art Supply Inventories, Media Releases, Copyright Registrations
            - "Memories": Documents that capture a fond memory. Examples: Letters, Notes, Theater Ticket Stubs, Photographs
            - "Other": Documents that don't fit neatly into other categories, or are unclear in purpose. Examples: Chinese Text Document, Abstract colorful image, Franklin Institute Color Codes, Roadside Attraction Sign

            Note: If the document is related to the catholic church, then it should fit the "Other" category and probably belongs to Colleen.

            Based on what you can see in this image, respond in JSON format:
             {
                 "description": "Brief descriptive title",
                 "category": "Your best category suggestion here",
                 "visible_text": "ALL text you can clearly read from the document (be thorough)",
                 "document_type": "What type of document this appears to be"
             }"""
            
            logging.info("=== VISION LLM ANALYSIS ===")
            logging.info(f"Vision Model: {self.vision_model}")
            logging.info(f"Analyzing image: {filename}")
            logging.info(f"File path: {file_path}")
            logging.debug(f"Full prompt sent to Vision LLM:\n{prompt}")
            
            # Prepare image path (convert TIFF to PNG if needed) and build file URL
            prepared_path = self._prepare_image_path(file_path)
            # Prefer embedding as a data URL to avoid file path resolution issues
            try:
                image_url = self._image_to_data_url(prepared_path)
            except Exception as encode_err:
                logging.warning(f"Failed to embed image as data URL, falling back to file URL: {encode_err}")
                image_url = self._make_file_url(prepared_path)

            # Also compute raw base64 for REST fallback
            b64_image = None
            try:
                b64_image = self._image_to_base64(prepared_path)
            except Exception as b64_err:
                logging.warning(f"Failed to base64 encode image for REST fallback: {b64_err}")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    # For Ollama in LangChain, image_url should be a string URL
                    {"type": "image_url", "image_url": image_url},
                ]
            )

            try:
                response = self.vision_llm.invoke([message])
                # Extract the text content from the response in a robust way
                response_text = None
                try:
                    # Some drivers may expose .text(); LangChain messages expose .content
                    if hasattr(response, "text") and callable(getattr(response, "text")):
                        response_text = response.text()
                    elif hasattr(response, "content"):
                        response_text = response.content
                    else:
                        response_text = str(response)
                except Exception:
                    response_text = str(response)
            except TypeError as type_err:
                # Common mismatch: ollama.Client.chat() missing 'images' keyword due to older ollama python pkg
                if "unexpected keyword argument 'images'" in str(type_err).lower():
                    logging.error("Ollama Python client appears outdated and doesn't support 'images' in chat(); attempting REST fallback.")
                    response_text = self._vision_chat_via_rest(prompt, b64_image)
                else:
                    raise

            logging.info(f"Vision LLM response: {response_text.strip() if isinstance(response_text, str) else response_text}")
            logging.info("=== END VISION LLM ANALYSIS ===")
        
            # Improved JSON extraction - find the first complete JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if not json_match:
                # Try a more permissive pattern for nested JSON
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    logging.debug(f"Successfully extracted JSON from vision analysis: {result}")
                    
                    # Extract date from filename or use creation date
                    date = self._extract_date("", filename, creation_date)
                    
                    # Get visible text from vision analysis
                    visible_text = result.get("visible_text", "").strip()
                    
                    # Use the visible text for identity detection if available; otherwise identity likely Unknown
                    identity = self._identify_person(visible_text) if visible_text else "Unknown"
                    
                    # Extract date from the visible text (or fallback to filename/creation date internally)
                    extracted_date = self._extract_date(visible_text or "", filename, creation_date)
                    
                    # Check if this is a paystub and apply consistent formatting
                    if self._is_paystub(visible_text, filename):
                        paystub_analysis = self._analyze_paystub_content(visible_text, filename)
                        return {
                            "identity": identity,
                            "date": extracted_date,
                            "description": paystub_analysis["description"],
                            "category": paystub_analysis["category"]
                        }
                    
                    # Return the LLM-provided description and category
                    return {
                        "identity": identity,
                        "date": extracted_date,
                        "description": result.get("description", "Image Document"),
                        "category": result.get("category", "Other")
                    }
                except json.JSONDecodeError as json_err:
                    logging.warning(f"JSON decode error for vision analysis of {filename}: {json_err}")
                    logging.debug(f"Problematic JSON string: {json_match.group(0)}")
                    # Fall through to fallback
            else:
                # Fallback if JSON parsing fails
                logging.warning(f"Could not parse LLM response for image {filename}")
                return self._create_image_fallback_result(filename, creation_date)
                
        except Exception as e:
            logging.error(f"Error in visual image analysis: {str(e)}")
            return self._create_image_fallback_result(filename, creation_date)

    def analyze_image_text(self, file_path: str, prompt: Optional[str] = None) -> str:
        """Extract text/description from an image using the vision model with a robust fallback.

        Returns a plain string response from the vision model.
        """
        if not self.vision_llm:
            logging.warning("Vision LLM not initialized, cannot perform image text analysis.")
            return ""

        default_prompt = (
            "Analyze this image carefully and extract ALL visible text (including small, stylized, or "
            "handwritten text). If readable text is minimal, describe what you can see in detail."
        )
        the_prompt = prompt or default_prompt

        try:
            prepared_path = self._prepare_image_path(file_path)
            # Prefer data URL; keep base64 for REST
            try:
                image_url = self._image_to_data_url(prepared_path)
            except Exception as encode_err:
                logging.warning(f"Failed to embed image as data URL, falling back to file URL: {encode_err}")
                image_url = self._make_file_url(prepared_path)

            b64_image = None
            try:
                b64_image = self._image_to_base64(prepared_path)
            except Exception as b64_err:
                logging.warning(f"Failed to base64 encode image for REST fallback: {b64_err}")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": the_prompt},
                    {"type": "image_url", "image_url": image_url},
                ]
            )

            try:
                response = self.vision_llm.invoke([message])
                response_text = self._as_text(response)
            except TypeError as type_err:
                if "unexpected keyword argument 'images'" in str(type_err).lower():
                    logging.error("Ollama client lacks 'images' support; using REST fallback for vision text.")
                    response_text = self._vision_chat_via_rest(the_prompt, b64_image)
                else:
                    raise

            if isinstance(response_text, str):
                return response_text.strip()
            return str(response_text)
        except Exception as e:
            logging.error(f"Error analyzing image for text: {e}")
            return ""

    def _vision_chat_via_rest(self, prompt: str, b64_image: Optional[str]) -> str:
        """Fallback path: call Ollama REST API /api/chat directly with base64 image.

        Returns the assistant message content as a string, or raises on HTTP error.
        """
        try:
            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        **({"images": [b64_image]} if b64_image else {}),
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    **({"top_p": 0.8, "top_k": 20, "min_p": 0.0} if self._is_qwen_model(self.vision_model) else {}),
                },
            }
            resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # shape: { "message": {"content": "..."}, ... }
            msg = data.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                # Some servers may return concatenated 'messages' or top-level 'content'
                content = data.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as rest_err:
            logging.error(f"REST vision fallback failed: {rest_err}")
            raise
    
    def _create_image_fallback_result(self, filename, creation_date):
        """Create a fallback result for images when visual analysis fails."""
        # Try to infer category from filename based on user-defined categories
        filename_lower = filename.lower()
        
        # Define category keywords with updated definitions (order matters - more specific first)
        if any(word in filename_lower for word in ['medical', 'prescription', 'lab', 'clinic', 'health', 'imaging', 'wellness']):
            category = "Medical"
        elif any(word in filename_lower for word in ['home', 'warranty', 'property', 'tax', 'permit', 'mortgage', 'closing', 'electricity', 'cable', 'utilities', 'maintenance', 'plant', 'plants', 'garden', 'gardening', 'flower', 'flowers', 'tree', 'trees', 'seed', 'seeds', 'fertilizer', 'soil', 'compost', 'landscaping', 'yard', 'lawn']):
            category = "Home"
        elif any(word in filename_lower for word in ['passport', 'driver', 'license', 'birth', 'social', 'vital']) or filename_lower in ['id.jpg', 'id.png', 'id.pdf', 'id.doc', 'id.docx'] or filename_lower.startswith('id_') or filename_lower.endswith('_id'):
            category = "Identification"
        elif any(word in filename_lower for word in ['car', 'auto', 'vehicle', 'registration', 'repair', 'title', 'claim']):
            category = "Auto"
        elif any(word in filename_lower for word in ['software', 'manual', 'warranty', 'network', 'config', 'technical', 'appliance', 'hardware', 'diagram', 'error']):
            category = "SysAdmin"
        elif any(word in filename_lower for word in ['degree', 'transcript', 'course', 'student', 'training', 'academic']):
            category = "School"
        elif any(word in filename_lower for word in ['recipe', 'cookbook', 'meal', 'plan', 'diet', 'culinary']):
            category = "Cooking"
        elif any(word in filename_lower for word in ['bank', 'statement', 'invoice', 'will', 'paystub', 'payment', 'investment']):
            category = "Financial"
        elif any(word in filename_lower for word in ['itinerary', 'ticket', 'boarding', 'hotel', 'trip', 'tourism', 'reservation', 'vacation']):
            category = "Travel"
        elif any(word in filename_lower for word in ['employment', 'contract', 'pay', 'benefit', 'review', 'history', 'performance']):
            category = "Employment"
        elif any(word in filename_lower for word in ['diy', 'craft', 'pattern', 'art', 'hobby', 'media', 'copyright']) or 'guide' in filename_lower:
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
            r"Charles\s+Collard", r"Chuck\s+Collard", r"Charles\s+W\.?\s+Collard",
            r"Charles\s+Colle", r"Chuck\s+Colle",  # Handle OCR truncation
            r"COLLARD,\s*CHARLES", r"Charles\s+W\s+Collard"  # Handle different formats
        ]
        
        colleen_patterns = [
            r"Colleen\s+McGinnis", r"Colleen\s+Collard", r"Colleen\s+Mueginnis",
            r"Colleen\s+McGinn", r"Colleen\s+Coll"  # Handle OCR truncation
        ]
        
        # Check for Chuck
        for pattern in chuck_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logging.debug(f"Identity detected as Chuck via pattern: {pattern}")
                return "Chuck"
        
        # Check for Colleen
        for pattern in colleen_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logging.debug(f"Identity detected as Colleen via pattern: {pattern}")
                return "Colleen"
        
        if not self.llm:
            logging.warning("LLM not initialized, returning 'Unknown' for identity.")
            return "Unknown"

        # Check if this is clearly a plant-related document
        plant_keywords = [
            'plant', 'plants', 'garden', 'gardening', 'flower', 'flowers', 'tree', 'trees',
            'seed', 'seeds', 'fertilizer', 'soil', 'compost', 'watering', 'pruning',
            'botanical', 'horticulture', 'landscaping', 'yard', 'lawn', 'shrub', 'shrubs',
            'bulb', 'bulbs', 'perennial', 'annual', 'greenhouse', 'nursery', 'planting'
        ]
        
        text_lower = text.lower()
        is_plant_related = any(keyword in text_lower for keyword in plant_keywords)
        
        if is_plant_related:
            logging.info("Document appears plant-related, assigning to Chuck")
            return "Chuck"

        # Check if this is clearly a church-related document
        church_keywords = [
            'church', 'catholic', 'diocese', 'parish', 'volunteer', 'ministry', 
            'religious', 'faith', 'congregation', 'liturgy', 'mass', 'prayer',
            'sacrament', 'confirmation', 'communion', 'baptism'
        ]
        
        is_church_related = any(keyword in text_lower for keyword in church_keywords)
        
        if is_church_related:
            logging.info("Document appears church-related, assigning to Colleen")
            return "Colleen"

        # If no match, use LLM to determine the most likely person
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Based on the following document text, determine if it most likely belongs to "Chuck Collard" 
            (also known as "Charles Collard" or "Charles W Collard") or "Colleen McGinnis" (also known as "Colleen Collard" or "Colleen Mueginnis").
            
            IMPORTANT RULES:
            - If the document is related to plants, gardening, flowers, trees, landscaping, or any botanical content, it belongs to Chuck.
            - If the document is related to the Catholic Church, religious activities, volunteering at church, 
              or contains terms like "diocese", "parish", "ministry", "faith", then it belongs to Colleen.
            - If the document contains any church-related content, always assign it to Colleen.
            - If the document contains hearing-related content, it likely belongs to Chuck.
            - Look for any variation of the names including different spellings.

            Document text:
            {text}
            
            Answer with only "Chuck" or "Colleen" or "Unknown":
            """
        )
        
        formatted_prompt = prompt.format(text=text[:5000])
        logging.info("=== LLM IDENTITY DETECTION ===")
        logging.info(f"Model: {self.model}")
        logging.info(f"Text preview: {text[:100]}...")
        logging.debug(f"Full prompt sent to LLM:\n{formatted_prompt}")

        response = self.llm.invoke(formatted_prompt)  # Use first 5000 chars for efficiency
        response_text = self._as_text(response)

        logging.info(f"LLM response: {response_text}")
        logging.info("=== END LLM IDENTITY DETECTION ===")
        
        if isinstance(response_text, str) and "chuck" in response_text.lower():
            logging.debug("Identity detected as Chuck via LLM")
            return "Chuck"
        elif isinstance(response_text, str) and "colleen" in response_text.lower():
            logging.debug("Identity detected as Colleen via LLM")
            return "Colleen"
        else:
            logging.debug("Identity could not be determined, returning Unknown")
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

        # First, check if this is a paystub - handle it specially for consistency
        if self._is_paystub(text, filename):
            return self._analyze_paystub_content(text, filename)

        prompt = PromptTemplate(
            input_variables=["text", "filename"],
            template="""
            Analyze the following document text and filename to:
            1. Create a brief descriptive title (5 words or less)
            2. Suggest the BEST category name for this document using one of the defined categories below:
            
            - "Medical": Documents related to personal and family health, including prescriptions, exam results, insurance information, and wellness records. Examples: Medical Imaging Reports, Lab Results, Physical Therapy Plans, Dulera Prescription, Eye Exam Prescription, Pupil Distance Waiver Form
            - "Identification": Passports, driver's licenses, IDs, and vital records. Examples: Passport, Driver's License, Birth Certificate, Social Security Card
            - "Home": Documents related to your residence, including purchase agreements, maintenance records, utilities, property information, and plant/gardening activities. Examples: Home Warranty, Property Tax Documents, Construction Permits, Mortgage Papers, Closing Documents, Homeowner Insurance, Electricity Bills, Cable Bills, Plant Care Guides, Garden Plans, Landscaping Documents
            - "Auto": Car titles, maintenance records, and vehicle-related paperwork. Examples: Car Title, Auto Repair Records, Registration Documents, Insurance Claim Forms, BMW Warranty Extension Details
            - "SysAdmin": Documents related to software, network configurations, and technical instructions, including Software licenses, user manuals, and tech warranties. Examples: Software Licenses, Hardware Specifications, Appliance Manuals, Product Warranties, Network Configuration Diagram, Technical Error Report
            - "School": Degrees, transcripts, and academic records. Examples: Degree Certificates, Transcripts, Course Materials, Student Loans Documents, FranklinCovey Training Notes
            - "Cooking": Collection of recipes, cookbooks, meal plans and related culinary information. Examples: Apple Raisin Crisp Recipe, Cooking Recipes, Meal Plans, Diet Guides
            - "Financial": Documents related to income, expenses, investments, and taxes. Examples: W-2s, Wills, Tax Documents, Bank Statements, Investment Records (NOTE: Pay stubs should be categorized as Employment, not Financial)
            - "Travel": Documents related to trips, vacations, and recreational activities. Examples: Travel Itineraries, Boarding Passes, Hotel Confirmation, Tourism Information, Trip Insurance
            - "Employment": Documents related to employment history, benefits, and income from work. Examples: Pay Stubs, Employment Contracts, Benefits Forms, Performance Reviews (NOTE: This is the correct category for paystubs and earnings statements)
            - "Hobbies": Documents related to personal hobbies and interests. Examples: DIY Guides, Craft Patterns, Art Supply Inventories, Media Releases, Copyright Registrations
            - "Memories": Documents that capture a fond memory. Examples: Letters, Notes, Theater Ticket Stubs, Photographs
            - "Other": Documents that don't fit neatly into other categories, or are unclear in purpose. Examples: Chinese Text Document, Abstract colorful image, Franklin Institute Color Codes, Roadside Attraction Sign.

            Note: If the document is related to church, then it should fit the "Other" category and probably belongs to Colleen.

            Document filename: {filename}
            Document text (partial):
            {text}
            
            Respond in JSON format:
            {{"description": "Brief title here", "category": "One of the categories above"}}
            """
        )
        
        try:
            formatted_prompt = prompt.format(text=text[:2000], filename=filename)
            logging.info("=== LLM DOCUMENT ANALYSIS ===")
            logging.info(f"Model: {self.model}")
            logging.info(f"Analyzing: {filename}")
            logging.info(f"Text preview: {text[:150]}...")
            logging.debug(f"Full prompt sent to LLM:\n{formatted_prompt}")
            
            response = self.llm.invoke(formatted_prompt)
            response_text = self._as_text(response)
            
            logging.info(f"LLM response: {response_text}")
            logging.info("=== END LLM DOCUMENT ANALYSIS ===")
            
            # Try to extract JSON from response
            logging.debug(f"Extracting JSON from LLM response for {filename}")
            
            # Improved JSON extraction - find the first complete JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if not json_match:
                # Try a more permissive pattern for nested JSON
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    logging.debug(f"Successfully extracted JSON: {result}")
                    return {
                        "description": result.get("description", "Unknown Document"),
                        "category": result.get("category", "Uncategorized")
                    }
                except json.JSONDecodeError as json_err:
                    logging.warning(f"JSON decode error for {filename}: {json_err}")
                    logging.debug(f"Problematic JSON string: {json_match.group(0)}")
                    # Fall through to fallback parsing
            
            # Fallback parsing if JSON extraction fails
            description_match = re.search(r'"description":\s*"([^"]+)"', response_text)
            category_match = re.search(r'"category":\s*"([^"]+)"', response_text)

            return {
                "description": description_match.group(1) if description_match else "Unknown Document",
                "category": category_match.group(1) if category_match else "Uncategorized"
            }
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return {"description": "Unknown Document", "category": "Uncategorized"}
    
    def _is_paystub(self, text, filename):
        """Check if the document is a paystub based on text content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # First, check for exclusions - documents that are definitely NOT paystubs
        exclusion_patterns = [
            # Tax documents
            'tax document', 'tax return', '1040', 'w-2', 'w2', 'form 1099', '1099',
            'schedule a', 'schedule b', 'schedule c', 'schedule d', 'schedule e',
            'tax preparation', 'tax filing', 'irs', 'internal revenue',
            
            # Receipts and invoices
            'receipt', 'invoice', 'bill', 'purchase', 'sale', 'transaction',
            'store receipt', 'credit card', 'debit card', 'payment receipt',
            'tire', 'tires', 'bmw', 'auto repair', 'service receipt',
            
            # Bank and financial statements
            'bank statement', 'account statement', 'monthly statement',
            'credit statement', 'loan statement', 'mortgage statement',
            'investment statement', 'brokerage statement',
            
            # Other financial documents
            'insurance policy', 'insurance claim', 'insurance premium',
            'property tax', 'real estate', 'deed', 'title',
            'will', 'estate', 'trust', 'beneficiary'
        ]
        
        # Check if this document matches exclusion patterns
        for pattern in exclusion_patterns:
            if pattern in text_lower or pattern in filename_lower:
                logging.info(f"Document excluded from paystub detection via pattern: {pattern}")
                return False
        
        # Paystub keywords that strongly indicate this is a paystub
        paystub_keywords = [
            'pay stub', 'paystub', 'payroll stub', 'earnings statement',
            'salary statement', 'wage statement', 'pay statement',
            'payroll statement', 'earnings and deductions statement'
        ]
        
        # Check for explicit paystub keywords in text or filename
        for keyword in paystub_keywords:
            if keyword in text_lower or keyword in filename_lower:
                logging.info(f"Paystub detected via explicit keyword: {keyword}")
                return True
        
        # Core paystub indicators that are specific to paystubs
        # These are terms that typically appear together on paystubs but not on other documents
        core_paystub_indicators = [
            'gross pay', 'net pay', 'current net pay', 'total current net pay',
            'pay period', 'bi-weekly', 'pay frequency', 'period ending',
            'hours worked', 'regular hours', 'overtime hours',
            'hourly rate', 'salary rate', 'current earnings', 'ytd earnings',
            'direct deposit', 'pay date', 'payroll date',
            'employee id', 'employee number', 'emp id', 'emp no'
        ]
        
        # Paystub-specific deduction indicators
        paystub_deduction_indicators = [
            'pretax deductions', 'after-tax deductions', 'voluntary deductions',
            'health insurance deduction', 'dental insurance deduction',
            '401k deduction', 'retirement deduction', 'pension deduction',
            'life insurance deduction', 'disability deduction'
        ]
        
        # Tax withholding indicators (more specific to paystubs)
        paystub_tax_indicators = [
            'fed withholding', 'federal withholding', 'fed inc tax withheld',
            'state withholding', 'state inc tax withheld', 'fica withheld',
            'medicare withheld', 'so sec withheld', 'social security withheld',
            'tax withholdings', 'payroll taxes'
        ]
        
        # Count indicators by category
        core_count = sum(1 for indicator in core_paystub_indicators 
                        if indicator in text_lower)
        deduction_count = sum(1 for indicator in paystub_deduction_indicators 
                             if indicator in text_lower)
        tax_count = sum(1 for indicator in paystub_tax_indicators 
                       if indicator in text_lower)
        
        # More restrictive criteria: need strong evidence across multiple categories
        total_indicators = core_count + deduction_count + tax_count
        
        # Require at least 2 core indicators AND (1 deduction OR 1 tax indicator)
        if core_count >= 2 and (deduction_count >= 1 or tax_count >= 1):
            logging.info(f"Paystub detected via strong indicators: core={core_count}, deductions={deduction_count}, taxes={tax_count}")
            return True
        
        # Alternative: if we have many total indicators (5+), it's likely a paystub
        # But only if we have at least 1 core indicator
        if total_indicators >= 5 and core_count >= 1:
            logging.info(f"Paystub detected via high indicator count: total={total_indicators}, core={core_count}")
            return True
        
        # Special case: if filename starts with A followed by numbers (your paystub naming pattern)
        # and contains specific paystub terms (not just any financial terms)
        if (re.match(r'^A\d+\.pdf$', filename, re.IGNORECASE) and 
            any(term in text_lower for term in ['net pay', 'gross pay', 'pay period', 'hours worked', 'payroll'])):
            logging.info(f"Paystub detected via filename pattern and specific paystub terms: {filename}")
            return True
            
        logging.debug(f"Document not classified as paystub: core={core_count}, deductions={deduction_count}, taxes={tax_count}")
        return False
    
    def _analyze_paystub_content(self, text, filename):
        """Analyze paystub content with consistent formatting."""
        logging.info(f"Analyzing paystub: {filename}")
        
        # Extract paystub number if present
        paystub_number = self._extract_paystub_number(text, filename)
        
        # Create standardized description
        if paystub_number:
            description = f"Pay Stub {paystub_number}"
        else:
            description = "Pay Stub"
        
        # Paystubs are always Employment category (not Financial)
        # This ensures consistency - paystubs are about employment income, not financial planning
        return {
            "description": description,
            "category": "Employment"
        }
    
    def _extract_paystub_number(self, text, filename):
        """Extract paystub number from text or filename."""
        # Look for common paystub number patterns
        patterns = [
            # Check filename first for A-number pattern (most reliable)
            r'^(A\d{4,8})\.pdf$',  # A12345.pdf format in filename
            # Then check text content
            r'(?:stub|check)[\s#]*([A-Z]?\d{4,8})',  # stub#12345, check A12345
            r'(?:number|no\.?)[\s#]*([A-Z]?\d{4,8})', # number: A12345
            r'\b([A-Z]\d{4,8})\b',  # A12345 format (word boundary)
            r'\b(\d{5,8})\b',       # 5-8 digit numbers (word boundary)
        ]
        
        # Check filename first (most reliable for your A-number format)
        if re.match(r'^A\d+\.pdf$', filename, re.IGNORECASE):
            match = re.match(r'^(A\d+)\.pdf$', filename, re.IGNORECASE)
            if match:
                number = match.group(1).upper()
                logging.debug(f"Extracted paystub number from filename: {number}")
                return number
        
        # Then check text content
        combined_text = f"{text} {filename}"
        
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                # Return the first reasonable match
                for match in matches:
                    if len(match) >= 4:  # Reasonable paystub number length
                        number = match.upper()
                        logging.debug(f"Extracted paystub number from text: {number}")
                        return number
        
        return None

    def _create_default_result(self, filename, creation_date):
        """Create a default result when analysis fails."""
        return {
            "identity": "Unknown",
            "date": creation_date.strftime('%Y-%m-%d'),
            "description": os.path.splitext(os.path.basename(filename))[0],
            "category": "Uncategorized"
        }
