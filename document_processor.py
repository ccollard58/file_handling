import os
import pytesseract
from PIL import Image
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self):
        # Configure pytesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text(self, file_path):
        """Extract text from PDF or image file."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif']:
                return self._extract_text_from_image(file_path)
            else:
                logging.warning(f"Unsupported file format: {file_ext}")
                return ""
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def _extract_text_from_pdf(self, file_path):
        """Extract text from PDF file."""
        text = ""
        try:
            pdf = PdfReader(file_path)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
              # If no text was extracted, try OCR
            if not text.strip():
                logging.info(f"No text found in PDF, attempting OCR: {file_path}")
                text = self._perform_ocr_on_pdf(file_path)
                
            return text
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_text_from_image(self, file_path):
        """Extract text from image file using OCR."""
        try:
            image = Image.open(file_path)

            # Handle specific image formats
            if image.format == 'GIF':
                image = image.convert('RGB')
            
            # Special handling for TIFF images, which can sometimes cause OCR issues
            if file_path.lower().endswith(('.tif', '.tiff')):
                # Convert to RGB if needed
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                # Log the TIFF processing
                logging.info(f"Processing TIFF image: {file_path}, mode: {image.mode}, size: {image.size}")
                
                # Some TIFF files might benefit from image enhancement for OCR
                # We'll keep it simple but effective
                try:
                    from PIL import ImageEnhance
                    # Increase contrast slightly to help OCR
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)  # Boost contrast by 50%
                except Exception as e:
                    logging.warning(f"Could not enhance TIFF image: {str(e)}")

            text = pytesseract.image_to_string(image)
            
            # For image files, add logging to help diagnose OCR issues
            if text.strip():
                text_preview = text.strip()[:50] + ('...' if len(text.strip()) > 50 else '')
                logging.info(f"OCR extracted {len(text.strip())} chars from {file_path}: {text_preview}")
            else:
                logging.info(f"OCR extracted no text from {file_path}")
                
            return text
        except Exception as e:
            logging.error(f"Error processing image {file_path}: {str(e)}")
            return ""
    
    def _perform_ocr_on_pdf(self, file_path):
        """Perform OCR on a PDF file."""
        # This is a simplified implementation
        # In a production system, you would convert each page to an image and OCR it
        try:
            # Convert first page to image and OCR it
            # This is just a placeholder - you'd need to convert PDF pages to images first
            return "OCR text from PDF would go here"
        except Exception as e:
            logging.error(f"Error performing OCR on PDF {file_path}: {str(e)}")
            return ""
