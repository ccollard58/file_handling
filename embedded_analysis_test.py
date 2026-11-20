import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import fitz
from PIL import Image, ImageDraw, ImageFont

from embedded_analysis import ANALYSIS_NAME, embed_analysis
from file_handler import FileHandler
from gui_simplified import FileAnalysisThread
from llm_analyzer import LLMAnalyzer


ANALYSIS = {
    "identity": "Test",
    "date": "2024-01-01",
    "description": "Example",
    "category": "Other",
    "model_response": "Model description with details",
}


def create_scanned_pdf(path, rotation=0):
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 65)
    draw.text((120, 180), "SEARCHABLE DOCUMENT", font=font, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with fitz.open() as document:
        page = document.new_page(width=600, height=400)
        page.insert_image(page.rect, stream=buffer.getvalue())
        page.set_rotation(rotation)
        document.save(path)


class EmbeddedAnalysisTests(unittest.TestCase):
    def test_analysis_thread_retains_results_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.png")
            Image.new("RGB", (10, 10), "white").save(path)
            with open(path, "rb") as stream:
                original = stream.read()
            processor = Mock()
            processor.extract_text.return_value = "Extracted text"
            analyzer = Mock()
            analyzer.analyze_document.return_value = ANALYSIS
            handler = FileHandler(os.path.join(directory, "output"))
            thread = FileAnalysisThread([path], processor, analyzer, handler)
            results = []
            errors = []
            thread.file_analyzed.connect(results.append)
            thread.analysis_error.connect(lambda _path, error: errors.append(error))
            thread.run()
            self.assertEqual(errors, [])
            self.assertEqual(results[0]["extracted_text"], "Extracted text")
            self.assertEqual(results[0]["model_analysis"], ANALYSIS)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), original)

    def test_vision_response_is_not_discarded(self):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.vision_llm = object()
        analyzer.vision_model = "test"
        response = json.dumps({
            "description": "Receipt",
            "category": "Financial",
            "visible_text": "Visible details",
            "document_type": "Receipt",
        })
        analyzer._invoke_vision_llm = Mock(return_value=response)
        analyzer._extract_date = Mock(return_value="2024-01-01")
        analyzer._identify_person = Mock(return_value="Unknown")
        analyzer._is_paystub = Mock(return_value=False)
        result = analyzer._analyze_image_visually("image.png", "image.png", datetime.now())
        self.assertEqual(result["model_response"], response)
        self.assertEqual(result["vision_analysis"]["visible_text"], "Visible details")

        analyzer._invoke_vision_llm.return_value = "Unstructured vision details"
        result = analyzer._analyze_image_visually("image.png", "image.png", datetime.now())
        self.assertEqual(result["model_response"], "Unstructured vision details")

    def test_scanned_pdf_is_searchable_and_analysis_is_embedded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "scan.pdf")
            create_scanned_pdf(path, rotation=90)
            with fitz.open(path) as original:
                before = original[0].get_pixmap().samples
            text = "Vision description\n" + "long text " * 5000
            embed_analysis(path, ANALYSIS, text)
            with fitz.open(path) as document:
                payload = json.loads(document.embfile_get(ANALYSIS_NAME))
                self.assertEqual(payload["analysis"], ANALYSIS)
                self.assertEqual(payload["extracted_text"], text)
                self.assertIn("SEARCHABLE", document[0].get_text())
                match = document[0].search_for("SEARCHABLE")[0]
                self.assertAlmostEqual(match.x0, 60, delta=5)
                self.assertAlmostEqual(match.y0, 90, delta=15)
                self.assertEqual(document[0].rotation, 90)
                self.assertEqual(document[0].get_pixmap().samples, before)

    def test_native_pdf_text_and_existing_attachments_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "native.pdf")
            with fitz.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Existing text")
                document.embfile_add("other.txt", b"existing")
                document.save(path)
            with patch("embedded_analysis.pytesseract.image_to_pdf_or_hocr") as ocr:
                embed_analysis(path, ANALYSIS, "Existing text")
                ocr.assert_not_called()
            embed_analysis(path, ANALYSIS, "Updated text")
            with fitz.open(path) as document:
                self.assertEqual(document[0].get_text().strip(), "Existing text")
                self.assertEqual(document.embfile_get("other.txt"), b"existing")
                self.assertEqual(
                    json.loads(document.embfile_get(ANALYSIS_NAME))["extracted_text"],
                    "Existing text",
                )
                payload = json.loads(document.embfile_get("document-analysis-2.json"))
                self.assertEqual(payload["extracted_text"], "Updated text")

    def test_images_preserve_pixels_and_unicode_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in (".jpg", ".png"):
                with self.subTest(extension=extension):
                    path = os.path.join(directory, "image" + extension)
                    Image.new("RGB", (32, 32), "blue").save(path)
                    with Image.open(path) as image:
                        before = image.tobytes()
                    embed_analysis(path, ANALYSIS, "Text café\nsecond line")
                    with Image.open(path) as image:
                        self.assertEqual(image.tobytes(), before)
                        if extension == ".jpg":
                            comment = image.getexif()[37510]
                            payload = json.loads(comment[8:].decode("utf-16-be"))
                        else:
                            payload = json.loads(image.text[ANALYSIS_NAME])
                        self.assertEqual(payload["extracted_text"], "Text café\nsecond line")

    def test_png_reprocessing_preserves_existing_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.png")
            Image.new("RGB", (10, 10), "white").save(path)
            with open(path, "rb") as stream:
                original = stream.read()
            embed_analysis(path, ANALYSIS, "first")
            embed_analysis(path, ANALYSIS, "second")
            with Image.open(path) as image:
                self.assertEqual(
                    json.loads(image.text[ANALYSIS_NAME])["extracted_text"], "first"
                )
                self.assertEqual(
                    json.loads(image.text["document-analysis-2.json"])["extracted_text"],
                    "second",
                )
            with open(path, "rb") as stream:
                updated = stream.read()
            self.assertTrue(updated.startswith(original[:-12]))
            self.assertTrue(updated.endswith(original[-12:]))

    def test_failure_leaves_source_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "scan.pdf")
            create_scanned_pdf(source)
            with open(source, "rb") as stream:
                original = stream.read()
            handler = FileHandler(os.path.join(directory, "output"))
            with patch(
                "embedded_analysis.pytesseract.image_to_pdf_or_hocr",
                side_effect=RuntimeError("OCR unavailable"),
            ):
                result = handler.move_and_rename_file(
                    source, "processed.pdf", os.path.join(directory, "output", "Other"),
                    ANALYSIS, "text",
                )
            self.assertIsNone(result)
            self.assertIn("OCR unavailable", handler.last_error)
            with open(source, "rb") as stream:
                self.assertEqual(stream.read(), original)
            self.assertFalse(os.path.exists(os.path.join(directory, "output", "Other", "processed.pdf")))
            self.assertEqual(os.listdir(handler.backup_dir), [])

    def test_move_embeds_analysis_and_keeps_original_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "scan.pdf")
            create_scanned_pdf(source)
            with open(source, "rb") as stream:
                original = stream.read()
            handler = FileHandler(os.path.join(directory, "output"))
            result = handler.move_and_rename_file(
                source, "processed.pdf", os.path.join(directory, "output", "Other"),
                ANALYSIS, "complete extracted text",
            )
            self.assertIsNotNone(result, handler.last_error)
            self.assertFalse(os.path.exists(source))
            with fitz.open(result) as document:
                payload = json.loads(document.embfile_get(ANALYSIS_NAME))
                self.assertEqual(payload["extracted_text"], "complete extracted text")
                self.assertEqual(payload["analysis"]["model_response"], ANALYSIS["model_response"])
                self.assertIn("SEARCHABLE", document[0].get_text())
            actions = handler._load_actions()
            with open(actions[0]["backup_path"], "rb") as stream:
                self.assertEqual(stream.read(), original)

    def test_jpeg_metadata_limit_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "image.jpg")
            Image.new("RGB", (10, 10), "white").save(source)
            handler = FileHandler(os.path.join(directory, "output"))
            result = handler.move_and_rename_file(
                source, "processed.jpg", os.path.join(directory, "output", "Other"),
                ANALYSIS, "x" * 40000,
            )
            self.assertIsNone(result)
            self.assertIn("too large", handler.last_error)
            self.assertTrue(os.path.exists(source))


if __name__ == "__main__":
    unittest.main()
