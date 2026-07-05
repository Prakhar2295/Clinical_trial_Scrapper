from docling.document_converter import DocumentConverter

from app.core.chunker import PAGE_BREAK


class PDFExtractor:
    """Wraps Docling to convert PDFs to markdown, with page-level slicing."""

    def __init__(self):
        self._converter = DocumentConverter()

    def _convert(self, file_path: str):
        return self._converter.convert(file_path)

    def extract_markdown(self, file_path: str) -> str:
        """Full-document markdown export, with page break markers preserved."""
        result = self._convert(file_path)
        return result.document.export_to_markdown(page_break_placeholder=PAGE_BREAK)

    def extract_pages(self, file_path: str, max_pages: int | None = None) -> str:
        """Markdown export limited to the first `max_pages` pages."""
        markdown = self.extract_markdown(file_path)
        if max_pages is None:
            return markdown
        pages = markdown.split(PAGE_BREAK)
        return PAGE_BREAK.join(pages[:max_pages])
