import json
import os
import logging
import re
from datetime import datetime

class CorrectionHandler:
    def __init__(self, corrections_file=None):
        if corrections_file is None:
            self.corrections_file = os.path.join(os.path.expanduser("~"), ".document_organizer_corrections.json")
        else:
            self.corrections_file = corrections_file
        self.corrections = self.load_corrections()

    def load_corrections(self):
        if os.path.exists(self.corrections_file):
            try:
                with open(self.corrections_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading corrections: {e}")
                return []
        return []

    def save_correction(self, original_filename, extracted_text, corrected_category, corrected_identity, original_category, original_identity):
        # Avoid duplicates if the exact same correction exists
        for c in self.corrections:
            if (c.get('filename') == original_filename and 
                c.get('corrected_category') == corrected_category and 
                c.get('corrected_identity') == corrected_identity):
                return

        correction = {
            "timestamp": datetime.now().isoformat(),
            "filename": original_filename,
            "text_snippet": extracted_text[:2000] if extracted_text else "", # Store first 2000 chars
            "corrected_category": corrected_category,
            "corrected_identity": corrected_identity,
            "original_category": original_category,
            "original_identity": original_identity
        }
        self.corrections.append(correction)
        self._save_to_file()
        logging.info(f"Saved correction for {original_filename}")

    def _save_to_file(self):
        try:
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving corrections: {e}")

    def get_relevant_corrections(self, filename, text, limit=5):
        """
        Find relevant corrections to use as examples.
        Strategy:
        1. Exact filename match (highest priority)
        2. Filename keyword match
        3. Recent corrections
        """
        valid_corrections = [c for c in self.corrections if c.get('corrected_category') or c.get('corrected_identity')]
        
        # 1. Exact filename match (excluding extension)
        base_name = os.path.splitext(filename)[0].lower()
        exact_matches = [c for c in valid_corrections if os.path.splitext(c.get('filename', ''))[0].lower() == base_name]
        if exact_matches:
            return exact_matches[:limit]

        # 2. Keyword match in filename
        # Split filename into words
        words = set(re.findall(r'\w+', base_name))
        scored_corrections = []
        for c in valid_corrections:
            c_name = os.path.splitext(c.get('filename', ''))[0].lower()
            c_words = set(re.findall(r'\w+', c_name))
            common = words.intersection(c_words)
            if common:
                scored_corrections.append((len(common), c))
        
        scored_corrections.sort(key=lambda x: x[0], reverse=True)
        if scored_corrections:
            return [x[1] for x in scored_corrections[:limit]]

        # 3. Fallback to recent corrections
        valid_corrections.sort(key=lambda x: x['timestamp'], reverse=True)
        return valid_corrections[:limit]
