"""Quick smoke test for chunk merge logic — run directly."""

import sys

sys.path.insert(0, "src")

from pathlib import Path

from citewise.ingest.chunk import chunk_page
from citewise.ingest.clean import clean_page

FIXTURES = Path("tests/fixtures")

page = clean_page(FIXTURES / "guide_with_parts.json")
assert page is not None
chunks = chunk_page(page)
urls = {c.url for c in chunks}
assert any("overview" in u for u in urls), f"overview missing from {urls}"
assert any("what-you-ll-get" in u for u in urls), f"what-you-ll-get missing from {urls}"

page2 = clean_page(FIXTURES / "guide_with_parts.json")
assert page2 is not None
ids1 = [c.chunk_id for c in chunk_page(page2)]
ids2 = [c.chunk_id for c in chunk_page(page2)]
assert ids1 == ids2, "chunk IDs not stable"

for c in chunks:
    assert page.title in c.text, f"Title missing from {c.chunk_id}"

page3 = clean_page(FIXTURES / "answer_with_table.json")
assert page3 is not None
for c in chunk_page(page3, max_tokens=400):
    assert c.tokens <= 420, f"{c.chunk_id}: {c.tokens} tokens"

page4 = clean_page(FIXTURES / "very_short_page.json")
assert page4 is not None
short = chunk_page(page4)
assert len(short) == 1
assert "Blue Badge" in short[0].text

print("ALL ASSERTIONS PASS")
