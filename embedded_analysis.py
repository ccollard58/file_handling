"""Embed document analysis and searchable OCR in processed files."""

import io
import json
import logging
import os
import zlib

import fitz
import pytesseract
from PIL import Image


ANALYSIS_NAME = "document-analysis.json"


def embed_analysis(file_path, analysis_result, extracted_text):
    """Write analysis to supported file formats; leave other formats unchanged."""
    extension = os.path.splitext(file_path)[1].lower()
    payload = json.dumps(
        {"analysis": analysis_result, "extracted_text": extracted_text},
        ensure_ascii=False,
        indent=2,
    )
    if extension == ".pdf":
        _embed_pdf(file_path, payload)
    elif extension in (".jpg", ".jpeg"):
        _embed_jpeg(file_path, payload)
    elif extension == ".png":
        _embed_png(file_path, payload)


def _embed_pdf(file_path, payload):
    with fitz.open(file_path) as document:
        if document.needs_pass or document.is_encrypted:
            raise ValueError("Cannot embed analysis in an encrypted PDF.")
        if document.get_sigflags() > 0:
            raise ValueError("Cannot modify a signed PDF without invalidating its signature.")
        for page in document:
            if page.get_text("text").strip():
                continue
            rotation = page.rotation
            try:
                page.set_rotation(0)
                pixmap = page.get_pixmap(dpi=200, colorspace=fitz.csRGB, alpha=False)
                with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                    ocr_pdf = pytesseract.image_to_pdf_or_hocr(
                        image, extension="pdf", config="-c textonly_pdf=1"
                    )
                with fitz.open(stream=ocr_pdf, filetype="pdf") as layer:
                    if layer[0].get_text("text").strip():
                        page.show_pdf_page(page.rect, layer, 0)
                    else:
                        logging.warning(
                            "OCR found no searchable text on page %s of %s",
                            page.number + 1, file_path,
                        )
            finally:
                page.set_rotation(rotation)
        name = ANALYSIS_NAME
        existing_names = set(document.embfile_names())
        counter = 2
        while name in existing_names:
            name = f"document-analysis-{counter}.json"
            counter += 1
        document.embfile_add(
            name, payload.encode("utf-8"),
            filename=name,
            desc="Document analysis and extracted text",
        )
        output = document.tobytes(garbage=3, deflate=True)
    with open(file_path, "wb") as target:
        target.write(output)


def _embed_jpeg(file_path, payload):
    # Replace the EXIF APP1 segment without recompressing the image pixels.
    with Image.open(file_path) as image:
        exif = image.getexif()
        exif[37510] = b"UNICODE\x00" + payload.encode("utf-16-be")
        exif_bytes = exif.tobytes()
    if len(exif_bytes) + 2 > 65535:
        raise ValueError("Analysis is too large for JPEG EXIF metadata.")
    with open(file_path, "rb") as source:
        data = source.read()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Invalid JPEG file.")
    result = bytearray(data[:2])
    result.extend(b"\xff\xe1" + (len(exif_bytes) + 2).to_bytes(2, "big") + exif_bytes)
    offset = 2
    while offset < len(data):
        if data[offset] != 0xff or offset + 4 > len(data):
            raise ValueError("Invalid JPEG segment.")
        marker = data[offset + 1]
        if marker in (0xda, 0xd9):
            result.extend(data[offset:])
            break
        length = int.from_bytes(data[offset + 2:offset + 4], "big")
        end = offset + 2 + length
        if length < 2 or end > len(data):
            raise ValueError("Invalid JPEG segment length.")
        if marker != 0xe1 or not data[offset + 4:end].startswith(b"Exif\x00\x00"):
            result.extend(data[offset:end])
        offset = end
    else:
        raise ValueError("JPEG has no image data.")
    with open(file_path, "wb") as target:
        target.write(result)


def _embed_png(file_path, payload):
    with open(file_path, "rb") as source:
        data = source.read()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError("Invalid PNG file.")
    offset = len(signature)
    names = set()
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        end = offset + length + 12
        if end > len(data):
            raise ValueError("Invalid PNG chunk length.")
        kind = data[offset + 4:offset + 8]
        content = data[offset + 8:end - 4]
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            names.add(content.split(b"\x00", 1)[0])
        if kind == b"IEND":
            break
        offset = end
    else:
        raise ValueError("PNG has no end chunk.")
    name = ANALYSIS_NAME
    counter = 2
    while name.encode("ascii") in names:
        name = f"document-analysis-{counter}.json"
        counter += 1
    content = name.encode("ascii") + b"\x00\x00\x00\x00\x00" + payload.encode("utf-8")
    chunk_type = b"iTXt"
    chunk = (
        len(content).to_bytes(4, "big") + chunk_type + content
        + (zlib.crc32(chunk_type + content) & 0xffffffff).to_bytes(4, "big")
    )
    with open(file_path, "wb") as target:
        target.write(data[:offset] + chunk + data[offset:])
