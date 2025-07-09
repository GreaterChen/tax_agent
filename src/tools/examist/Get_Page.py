"""
Book_Tool.py
-------------
A small utility to extract specific pages (or page ranges) from a textbook-like
plain‑text file using explicit <page_n/start> … <page_n/end> delimiters.

Change log
----------
* 2025‑07‑01  Add trailing “End of Requested Content.” line (requested by user).
* 2025‑07‑02  Simplify API: Get_Page now accepts a list of page-range strings directly.

Usage example
-------------
>>> from Book_Tool import Get_Page
>>> print(Get_Page(["12-14", "15-20"]))

Requested Content from Textbook:
Page 12-20
Page 12:
<content of page 12>
...
End of Requested Content.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Configuration – adjust the textbook path here if necessary
# ---------------------------------------------------------------------------
TEXTBOOK_PATH = Path("src/tools/examist/Dependencies/Books/Book.txt")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_requested_ranges(range_list: List[str]) -> List[int]:
    """Convert a list of page-range strings (e.g. ["1-3", "5", "7-9"]) into a
    sorted list of *unique* page numbers, with overlaps/duplicates removed."""
    pages: List[int] = []
    for item in range_list:
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start  # tolerate reversed ranges
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(item))
    return sorted(set(pages))


def _merge_consecutive_pages(page_numbers: List[int]) -> List[Tuple[int, int]]:
    """Merge consecutive numbers into (start, end) tuples."""
    if not page_numbers:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = page_numbers[0]
    for page in page_numbers[1:]:
        if page == prev + 1:
            prev = page
        else:
            ranges.append((start, prev))
            start = prev = page
    ranges.append((start, prev))
    return ranges


def _load_textbook(path: Path = TEXTBOOK_PATH) -> Dict[int, str]:
    """Parse the textbook file and return {page_number: page_content}."""
    if not path.exists():
        raise FileNotFoundError(f"Textbook file not found: {path.resolve()}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(r"<page_(\d+)/start>(.*?)<page_\1/end>", re.DOTALL | re.IGNORECASE)
    return {int(m.group(1)): m.group(2).strip() for m in pattern.finditer(text)}


def _format_output(page_numbers: List[int], page_contents: Dict[int, str]) -> str:
    """Return the final formatted string according to the required template."""
    if not page_numbers:
        return "**Relevant Content from Textbook**\n(No pages found)\n\n**End of Requested Textbook Content.**"

    merged_ranges = _merge_consecutive_pages(page_numbers)
    range_strings = [f"{s}-{e}" if s != e else f"{s}" for s, e in merged_ranges]
    header = "Relebant Content from Textbook**\n\nPage " + ", ".join(range_strings)

    body_lines: List[str] = []
    for page in page_numbers:
        body_lines.append(f"\nPage {page}:")
        body_lines.append(page_contents.get(page, "(Content not found)"))

    body = "\n".join(body_lines)
    return f"{header}\n{body}\n\n**End of the Relevant Content from the Textbook**"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def Get_Page(request_list: List[str]) -> str:
    """Primary entry‑point. Accepts a list of page-range strings like:
        ["12-14", "15", "18-20"]
    """
    if not isinstance(request_list, list):
        raise ValueError("Input must be a list of page-range strings.")

    requested_pages = _parse_requested_ranges(request_list)
    textbook_pages = _load_textbook()
    return _format_output(requested_pages, textbook_pages)

# ---------------------------------------------------------------------------
# Self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example = ["658-661"]
    print(Get_Page(example))
