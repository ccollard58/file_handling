import json
import os
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two text strings."""
    if not text1 or not text2:
        return 0.0
    # Use first 500 chars for efficiency
    return SequenceMatcher(None, text1[:500].lower(), text2[:500].lower()).ratio()


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text for matching."""
    if not text:
        return set()
    # Extract words, filter short ones and common stop words
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                  'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                  'will', 'your', 'from', 'they', 'this', 'that', 'with', 'which'}
    return words - stop_words


class CorrectionHandler:
    def __init__(self, corrections_file: Optional[str] = None):
        if corrections_file is None:
            self.corrections_file = os.path.join(os.path.expanduser("~"), ".document_organizer_corrections.json")
        else:
            self.corrections_file = corrections_file
        self.corrections = self.load_corrections()

    def load_corrections(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.corrections_file):
            try:
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Migrate old format corrections to new format
                    return [self._migrate_correction(c) for c in data]
            except Exception as e:
                logging.error(f"Error loading corrections: {e}")
                return []
        return []

    def _migrate_correction(self, correction: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old correction format to new format with all fields."""
        # Ensure all new fields exist with sensible defaults
        if 'corrected_description' not in correction:
            correction['corrected_description'] = None
        if 'original_description' not in correction:
            correction['original_description'] = None
        if 'corrected_new_filename' not in correction:
            correction['corrected_new_filename'] = None
        if 'original_new_filename' not in correction:
            correction['original_new_filename'] = None
        if 'corrected_destination_folder' not in correction:
            # Map from corrected_category if it looks like a folder path
            correction['corrected_destination_folder'] = correction.get('corrected_category')
        if 'original_destination_folder' not in correction:
            correction['original_destination_folder'] = correction.get('original_category')
        return correction

    def save_correction(
        self, 
        original_filename: str, 
        extracted_text: str,
        corrected_category: Optional[str] = None,
        corrected_identity: Optional[str] = None,
        original_category: Optional[str] = None,
        original_identity: Optional[str] = None,
        corrected_description: Optional[str] = None,
        original_description: Optional[str] = None,
        corrected_new_filename: Optional[str] = None,
        original_new_filename: Optional[str] = None,
        corrected_destination_folder: Optional[str] = None,
        original_destination_folder: Optional[str] = None
    ) -> bool:
        """
        Save a correction for future few-shot learning.
        
        Returns True if a new correction was saved, False if skipped (duplicate or no changes).
        """
        # Check if any actual corrections were made
        has_changes = any([
            corrected_category and corrected_category != original_category,
            corrected_identity and corrected_identity != original_identity,
            corrected_description and corrected_description != original_description,
            corrected_new_filename and corrected_new_filename != original_new_filename,
            corrected_destination_folder and corrected_destination_folder != original_destination_folder,
        ])
        
        if not has_changes:
            logging.debug(f"No changes detected for {original_filename}, skipping correction save")
            return False

        # Avoid duplicates - check if same file with same corrections exists
        for c in self.corrections:
            if (c.get('filename') == original_filename and 
                c.get('corrected_category') == corrected_category and 
                c.get('corrected_identity') == corrected_identity and
                c.get('corrected_description') == corrected_description and
                c.get('corrected_new_filename') == corrected_new_filename and
                c.get('corrected_destination_folder') == corrected_destination_folder):
                logging.debug(f"Duplicate correction for {original_filename}, skipping")
                return False

        correction = {
            "timestamp": datetime.now().isoformat(),
            "filename": original_filename,
            "text_snippet": extracted_text[:2000] if extracted_text else "",
            # Category corrections
            "corrected_category": corrected_category,
            "original_category": original_category,
            # Identity corrections
            "corrected_identity": corrected_identity,
            "original_identity": original_identity,
            # Description corrections
            "corrected_description": corrected_description,
            "original_description": original_description,
            # Filename corrections
            "corrected_new_filename": corrected_new_filename,
            "original_new_filename": original_new_filename,
            # Destination folder corrections
            "corrected_destination_folder": corrected_destination_folder,
            "original_destination_folder": original_destination_folder,
        }
        self.corrections.append(correction)
        self._save_to_file()
        logging.info(f"Saved correction for {original_filename}: category={corrected_category}, identity={corrected_identity}, description={corrected_description}")
        return True

    def _save_to_file(self) -> None:
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving corrections: {e}")

    def get_relevant_corrections(
        self, 
        filename: str, 
        text: str, 
        limit: int = 5,
        match_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find relevant corrections to use as few-shot examples.
        
        Args:
            filename: The filename being analyzed
            text: The extracted text from the document
            limit: Maximum number of corrections to return
            match_type: Optional filter for specific correction types 
                       ('category', 'identity', 'description', 'filename', 'destination')
        
        Returns:
            List of relevant corrections, scored and sorted by relevance
        """
        valid_corrections = [c for c in self.corrections if self._has_valid_corrections(c, match_type)]
        
        if not valid_corrections:
            return []

        # Score each correction
        scored_corrections = []
        base_name = os.path.splitext(filename)[0].lower()
        filename_words = _extract_keywords(base_name)
        text_keywords = _extract_keywords(text)
        
        for c in valid_corrections:
            score = self._score_correction(c, filename, base_name, filename_words, text, text_keywords)
            if score > 0:
                scored_corrections.append((score, c))
        
        # Sort by score descending
        scored_corrections.sort(key=lambda x: x[0], reverse=True)
        
        return [x[1] for x in scored_corrections[:limit]]

    def _has_valid_corrections(self, correction: Dict[str, Any], match_type: Optional[str] = None) -> bool:
        """Check if correction has valid corrected values for the specified type."""
        if match_type == 'category':
            return bool(correction.get('corrected_category'))
        elif match_type == 'identity':
            return bool(correction.get('corrected_identity') and correction['corrected_identity'] != 'Unknown')
        elif match_type == 'description':
            return bool(correction.get('corrected_description'))
        elif match_type == 'filename':
            return bool(correction.get('corrected_new_filename'))
        elif match_type == 'destination':
            return bool(correction.get('corrected_destination_folder'))
        else:
            # Any valid correction
            return any([
                correction.get('corrected_category'),
                correction.get('corrected_identity') and correction['corrected_identity'] != 'Unknown',
                correction.get('corrected_description'),
                correction.get('corrected_new_filename'),
                correction.get('corrected_destination_folder'),
            ])

    def _score_correction(
        self, 
        correction: Dict[str, Any],
        filename: str,
        base_name: str,
        filename_words: set,
        text: str,
        text_keywords: set
    ) -> float:
        """
        Score a correction based on similarity to current document.
        
        Higher scores indicate more relevant corrections.
        """
        score = 0.0
        
        c_filename = correction.get('filename', '')
        c_base_name = os.path.splitext(c_filename)[0].lower()
        c_text = correction.get('text_snippet', '')
        
        # 1. Exact filename match (highest priority)
        if c_base_name == base_name:
            score += 100.0
            return score  # Perfect match, return immediately
        
        # 2. Filename word overlap
        c_filename_words = _extract_keywords(c_base_name)
        if filename_words and c_filename_words:
            common_words = filename_words.intersection(c_filename_words)
            word_score = len(common_words) / max(len(filename_words), len(c_filename_words))
            score += word_score * 30.0
        
        # 3. Text content similarity
        if text and c_text:
            text_sim = _text_similarity(text, c_text)
            score += text_sim * 40.0
        
        # 4. Keyword overlap in text
        c_text_keywords = _extract_keywords(c_text)
        if text_keywords and c_text_keywords:
            common_keywords = text_keywords.intersection(c_text_keywords)
            if common_keywords:
                keyword_score = len(common_keywords) / max(len(text_keywords), len(c_text_keywords))
                score += keyword_score * 20.0
        
        # 5. Recency bonus (more recent corrections slightly preferred)
        try:
            timestamp = datetime.fromisoformat(correction.get('timestamp', ''))
            days_old = (datetime.now() - timestamp).days
            recency_score = max(0, 10 - days_old / 30)  # Decay over 300 days
            score += recency_score
        except:
            pass
        
        return score

    def get_few_shot_examples_for_category(self, filename: str, text: str, limit: int = 3) -> str:
        """
        Generate few-shot examples string for category classification.
        
        Returns a formatted string to include in the LLM prompt.
        """
        corrections = self.get_relevant_corrections(filename, text, limit=limit, match_type='category')
        
        if not corrections:
            return ""
        
        examples = []
        for c in corrections:
            example = f"""Example {len(examples) + 1}:
  Filename: {c.get('filename', 'Unknown')}
  Text snippet: {c.get('text_snippet', '')[:200]}...
  Original category: {c.get('original_category', 'Unknown')}
  CORRECTED to category: {c.get('corrected_category', 'Unknown')}"""
            if c.get('corrected_description'):
                example += f"\n  Description: {c.get('corrected_description')}"
            examples.append(example)
        
        return "\n\nHere are examples of how similar documents were correctly categorized:\n" + "\n\n".join(examples) + "\n\nUse these examples to guide your categorization.\n"

    def get_few_shot_examples_for_identity(self, filename: str, text: str, limit: int = 3) -> str:
        """
        Generate few-shot examples string for identity detection.
        
        Returns a formatted string to include in the LLM prompt.
        """
        corrections = self.get_relevant_corrections(filename, text, limit=limit, match_type='identity')
        
        if not corrections:
            return ""
        
        examples = []
        for c in corrections:
            example = f"""Example {len(examples) + 1}:
  Filename: {c.get('filename', 'Unknown')}
  Text snippet: {c.get('text_snippet', '')[:200]}...
  Original identity: {c.get('original_identity', 'Unknown')}
  CORRECTED to identity: {c.get('corrected_identity', 'Unknown')}"""
            examples.append(example)
        
        return "\n\nHere are examples of how similar documents were correctly identified:\n" + "\n\n".join(examples) + "\n\nUse these examples to determine the identity.\n"

    def get_few_shot_examples_for_description(self, filename: str, text: str, limit: int = 3) -> str:
        """
        Generate few-shot examples string for description generation.
        
        Returns a formatted string to include in the LLM prompt.
        """
        corrections = self.get_relevant_corrections(filename, text, limit=limit, match_type='description')
        
        if not corrections:
            return ""
        
        examples = []
        for c in corrections:
            example = f"""Example {len(examples) + 1}:
  Filename: {c.get('filename', 'Unknown')}
  Text snippet: {c.get('text_snippet', '')[:150]}...
  Original description: {c.get('original_description', 'Unknown')}
  CORRECTED to description: {c.get('corrected_description', 'Unknown')}"""
            examples.append(example)
        
        return "\n\nHere are examples of good document descriptions:\n" + "\n\n".join(examples) + "\n\nUse these examples to create a clear, concise description.\n"

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about saved corrections."""
        total = len(self.corrections)
        category_corrections = sum(1 for c in self.corrections if c.get('corrected_category') != c.get('original_category'))
        identity_corrections = sum(1 for c in self.corrections if c.get('corrected_identity') != c.get('original_identity'))
        description_corrections = sum(1 for c in self.corrections if c.get('corrected_description'))
        filename_corrections = sum(1 for c in self.corrections if c.get('corrected_new_filename'))
        
        return {
            'total_corrections': total,
            'category_corrections': category_corrections,
            'identity_corrections': identity_corrections,
            'description_corrections': description_corrections,
            'filename_corrections': filename_corrections,
        }

    def clear_corrections(self) -> None:
        """Clear all saved corrections."""
        self.corrections = []
        self._save_to_file()
        logging.info("Cleared all corrections")

    def export_corrections(self, export_path: str) -> bool:
        """Export corrections to a file."""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, indent=2, ensure_ascii=False)
            logging.info(f"Exported {len(self.corrections)} corrections to {export_path}")
            return True
        except Exception as e:
            logging.error(f"Error exporting corrections: {e}")
            return False

    def import_corrections(self, import_path: str, merge: bool = True) -> int:
        """
        Import corrections from a file.
        
        Args:
            import_path: Path to the JSON file to import
            merge: If True, merge with existing; if False, replace
            
        Returns:
            Number of corrections imported
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported = json.load(f)
            
            if not isinstance(imported, list):
                logging.error("Invalid corrections file format")
                return 0
            
            # Migrate imported corrections
            imported = [self._migrate_correction(c) for c in imported]
            
            if merge:
                # Add only non-duplicate corrections
                existing_keys = set()
                for c in self.corrections:
                    key = (c.get('filename'), c.get('corrected_category'), c.get('corrected_identity'))
                    existing_keys.add(key)
                
                added = 0
                for c in imported:
                    key = (c.get('filename'), c.get('corrected_category'), c.get('corrected_identity'))
                    if key not in existing_keys:
                        self.corrections.append(c)
                        existing_keys.add(key)
                        added += 1
                
                self._save_to_file()
                logging.info(f"Imported {added} new corrections (merged)")
                return added
            else:
                self.corrections = imported
                self._save_to_file()
                logging.info(f"Imported {len(imported)} corrections (replaced)")
                return len(imported)
                
        except Exception as e:
            logging.error(f"Error importing corrections: {e}")
            return 0
