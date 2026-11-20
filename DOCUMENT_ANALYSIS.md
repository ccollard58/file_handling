# Embedded document analysis

When selected files are processed, the application embeds the extracted text,
classification, and available model response in the resulting file. Analysis
alone does not modify the source. The existing backup contains the original
file before processing.

- PDF: open the `document-analysis.json` attachment in a PDF viewer that
  supports attachments. Reprocessing adds a numbered attachment instead of
  replacing existing attachments. Pages with no existing selectable text receive an
  invisible, positioned Tesseract OCR layer; OCR uses the local Tesseract
  installation. Pages that already have selectable text are not OCRed, even
  if they also contain scanned images. OCR may miss or misrecognize text, and
  the model's descriptive text is not used as positioned OCR.
- JPEG: analysis is stored in the EXIF UserComment field. JPEG EXIF has a
  size limit; processing fails rather than truncate analysis that exceeds it.
- PNG: analysis is stored in the `document-analysis.json` text metadata.
  Reprocessing adds a numbered text entry without replacing existing metadata.

Other formats are moved without embedded analysis. Encrypted or signed PDFs
are not modified. Processing reports embedding or OCR errors and leaves the
source in place. The embedded analysis may contain sensitive document text and
travels with the file.
