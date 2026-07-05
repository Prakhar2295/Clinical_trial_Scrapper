from langchain_text_splitters import MarkdownTextSplitter

PAGE_BREAK = "\n\n<!-- PAGE_BREAK -->\n\n"

CHUNK_SIZE_CHARS = 8000  # ~2000 tokens at ~4 chars/token for English clinical text
CHUNK_OVERLAP_CHARS = 800  # ~200 tokens

TOC_MARKER = "table of contents"
TOC_SEARCH_PAGES = 5
HEAD_PAGES_AFTER_TOC = 2
HEAD_PAGES_NO_TOC = 7
TAIL_PAGES = 3
TAIL_MIN_TOTAL_PAGES = 10


def split_pages(markdown_text: str) -> list[str]:
    """Split full-document markdown into per-page text, dropping blank pages."""
    return [page for page in markdown_text.split(PAGE_BREAK) if page.strip()]


def chunk_text(markdown: str) -> list[str]:
    """Split markdown into chunks via LangChain's MarkdownTextSplitter.

    Splits on markdown headings (##, ###, ####) first, falling back to
    character count, so section boundaries in clinical trial documents are
    preserved. chunk_size/overlap are in characters, not tokens (~4 chars per
    token for English clinical text): 8000/800 chars approximate 2000/200 tokens.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )
    chunks = splitter.split_text(markdown)
    return [c.strip() for c in chunks if c.strip()]


def _find_toc_page_index(pages: list[str]) -> int | None:
    """0-based index of the first page (within the first TOC_SEARCH_PAGES pages)
    that looks like a table of contents, or None if none is found."""
    for i, page in enumerate(pages[:TOC_SEARCH_PAGES]):
        if TOC_MARKER in page.lower():
            return i
    return None


def chunk_for_ingestion(markdown_text: str) -> list[dict]:
    """
    Bootstrap ingestion chunking:
      - head: TOC page + 2 following pages, if a table of contents is found
        in the first 5 pages; otherwise the first 7 pages.
      - tail: last 3 pages, only if the document has more than 10 pages,
        excluding any pages already covered by head.
    Returns a list of {content, chunk_position, chunk_index, page_range}.
    """
    pages = split_pages(markdown_text)
    total_pages = len(pages)
    results: list[dict] = []

    toc_idx = _find_toc_page_index(pages)
    if toc_idx is not None:
        head_page_count = min(toc_idx + 1 + HEAD_PAGES_AFTER_TOC, total_pages)
    else:
        head_page_count = min(HEAD_PAGES_NO_TOC, total_pages)

    head_text = "\n\n".join(pages[:head_page_count])
    for i, chunk in enumerate(chunk_text(head_text)):
        results.append(
            {
                "content": chunk,
                "chunk_position": "head",
                "chunk_index": i,
                "page_range": f"1-{head_page_count}",
            }
        )

    if total_pages > TAIL_MIN_TOTAL_PAGES:
        tail_start = max(total_pages - TAIL_PAGES, head_page_count)
        if tail_start < total_pages:
            tail_text = "\n\n".join(pages[tail_start:])
            start_index = len(results)
            for i, chunk in enumerate(chunk_text(tail_text)):
                results.append(
                    {
                        "content": chunk,
                        "chunk_position": "tail",
                        "chunk_index": start_index + i,
                        "page_range": f"{tail_start + 1}-{total_pages}",
                    }
                )

    return results


def chunk_for_inference(markdown_text: str, max_pages: int) -> list[dict]:
    """Chunk the first `max_pages` pages of an untagged document for classification."""
    pages = split_pages(markdown_text)
    page_text = "\n\n".join(pages[:max_pages])

    results = []
    for i, chunk in enumerate(chunk_text(page_text)):
        results.append(
            {
                "content": chunk,
                "chunk_position": "head",
                "chunk_index": i,
                "page_range": f"1-{min(max_pages, len(pages))}",
            }
        )
    return results
