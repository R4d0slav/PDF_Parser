"""
PDF → JSON Parser (powered by marker-pdf)
------------------------------------------
Extracts content from a PDF and saves it as structured JSON,
grouped by section. Text is always extracted. Use flags to
include tables, figures, and equations.

Usage:
    python pdf_parser.py <input.pdf> [options]

Examples:
    python pdf_parser.py report.pdf
    python pdf_parser.py report.pdf -o results.json --tables --figures
    python pdf_parser.py report.pdf --tables --figures --equations

Output format:
    {
        "source": "input.pdf",
        "metadata": { "pages": 15, "toc": [...] },
        "sections": [
            {
                "title": "3.2 Attention",
                "text": "An attention function can be described as...",
                "equations": ["$$Attention(Q,K,V) = softmax(...)V$$"],
                "tables": [{"caption": "Table 1: ...", "content": "| ... |"}],
                "figures": [
                    {
                        "caption": "Figure 1: The Transformer model architecture.",
                        "image_id": "_page_2_Figure_0.jpeg",
                        "image_path": "results/figures/_page_2_Figure_0.jpeg.png"
                    }
                ]
            }
        ]
    }

Dependencies:
    pip install marker-pdf

Notes:
    - Figures are saved as PNG files next to the output JSON
    - Image IDs match the keys in marker-pdf's rendered.images dict
    - Equations are LaTeX strings
    - Tables are raw Markdown table strings with captions
    - Headers/footers are automatically stripped by marker-pdf
"""

import os
import re
import json
import argparse
from pathlib import Path

os.environ["TORCH_DEVICE"] = "cpu"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_heading(line: str) -> bool:
    """Detect markdown headings or numbered section headings."""
    if re.match(r"^#{1,6}\s+\S", line):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True
    return False


def clean_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def is_noise(text: str) -> bool:
    if re.fullmatch(r"[\d\s]+", text):
        return True
    if len(text) < 15 and " " not in text:
        return True
    return False


def save_images(images: dict, images_dir: Path) -> dict:
    """Save PIL images to disk, return {image_id: file_path}."""
    images_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key, img in images.items():
        filename = re.sub(r"[^\w\-.]", "_", key) + ".png"
        filepath = images_dir / filename
        img.save(filepath, format="PNG")
        saved[key] = str(filepath)
    return saved


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_markdown(markdown: str, allowed_types: set, saved_images: dict) -> list:
    """
    Parse marker-pdf markdown into clean sections.

    Each section is a dict with:
        title     : str
        text      : str (all paragraphs merged)
        equations : list[str]
        tables    : list[{caption, content}]
        figures   : list[{caption, image_id, image_path}]

    Images are linked by matching ![](_page_X_Figure_Y.jpeg) references
    to their saved file paths and the following caption line.
    Table captions are linked to the table rows that follow them.
    Single-line and multi-line $$ equations are both handled correctly.
    """

    lines = markdown.split("\n")
    sections = []

    current = {
        "title": "Preamble",
        "text_lines": [],
        "equations": [],
        "tables": [],
        "figures": [],
        "pending_image_id": None,
        "pending_table_caption": None,
    }

    def flush_section():
        if current["pending_image_id"]:
            if "figure" in allowed_types:
                current["figures"].append({
                    "caption": "",
                    "image_id": current["pending_image_id"],
                    "image_path": saved_images.get(current["pending_image_id"], ""),
                })
        current["pending_image_id"] = None
        current["pending_table_caption"] = None

        merged_text = " ".join(current["text_lines"]).strip()
        merged_text = re.sub(r"\s+", " ", merged_text)

        section = {"title": current["title"]}
        if "text"     in allowed_types: section["text"]      = merged_text
        if "equation" in allowed_types: section["equations"] = current["equations"].copy()
        if "table"    in allowed_types: section["tables"]    = current["tables"].copy()
        if "figure"   in allowed_types: section["figures"]   = current["figures"].copy()

        if merged_text or current["equations"] or current["tables"] or current["figures"]:
            sections.append(section)

        current["text_lines"].clear()
        current["equations"].clear()
        current["tables"].clear()
        current["figures"].clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Section heading ──────────────────────────────────────────────────
        if is_heading(line):
            flush_section()
            current["title"] = clean_heading(line)
            current["pending_image_id"] = None
            current["pending_table_caption"] = None
            i += 1
            continue

        # ── Skip blank lines (preserve pending ids across them) ──────────────
        if not line.strip():
            i += 1
            continue

        # Strip inline HTML tags for pattern matching
        clean_line = re.sub(r"<[^>]+>", "", line).strip()

        # ── Image reference: ![](_page_X_Figure_Y.jpeg) ──────────────────────
        img_match = re.match(r"!\[.*?\]\((.+?)\)", line)
        if img_match:
            current["pending_image_id"] = img_match.group(1)
            i += 1
            continue

        # ── Figure caption: Figure N: ... ────────────────────────────────────
        if re.match(r"^(Figure|Fig\.)\s*\d+", clean_line, re.IGNORECASE):
            if "figure" in allowed_types:
                current["figures"].append({
                    "caption": clean_line,
                    "image_id": current["pending_image_id"] or "",
                    "image_path": saved_images.get(current["pending_image_id"], "") if current["pending_image_id"] else "",
                })
            current["pending_image_id"] = None
            i += 1
            continue

        # ── Table caption: Table N: ... ──────────────────────────────────────
        if re.match(r"^Table\s*\d+", clean_line, re.IGNORECASE):
            current["pending_table_caption"] = clean_line
            i += 1
            continue

        # ── Equation block: $$ ... $$ ─────────────────────────────────────────
        if line.strip().startswith("$$"):
            # Single-line equation: $$...$$ opens and closes on the same line
            if line.strip().endswith("$$") and len(line.strip()) > 4:
                if "equation" in allowed_types:
                    current["equations"].append(line.strip())
                i += 1
                continue
            # Multi-line equation: $$ alone on a line, collect until closing $$
            eq_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                eq_lines.append(lines[i])
                i += 1
            if i < len(lines):
                eq_lines.append(lines[i])
                i += 1
            if "equation" in allowed_types:
                current["equations"].append("\n".join(eq_lines).strip())
            continue

        # ── Table block: | ... | lines ────────────────────────────────────────
        if line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if "table" in allowed_types:
                current["tables"].append({
                    "caption": current["pending_table_caption"] or "",
                    "content": "\n".join(table_lines),
                })
            current["pending_table_caption"] = None
            continue

        # ── Regular text ──────────────────────────────────────────────────────
        if "text" in allowed_types:
            clean = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", line).strip()
            clean = re.sub(r"<[^>]+>", "", clean).strip()
            clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
            if clean and not is_noise(clean):
                current["text_lines"].append(clean)

        i += 1

    flush_section()
    return sections


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process(pdf_path: str, output_path: str, allowed_types: set) -> None:
    print("[1/3] Loading marker-pdf models...")
    converter = PdfConverter(artifact_dict=create_model_dict())

    print(f"[2/3] Converting '{pdf_path}'...")
    rendered = converter(pdf_path)

    # Save images before serializing (PIL images aren't JSON serializable)
    raw_images = rendered.images or {}
    images_dir = Path(output_path).with_suffix("") / "figures"
    saved_images = {}
    if raw_images and "figure" in allowed_types:
        print(f"      Saving {len(raw_images)} figure image(s)...")
        saved_images = save_images(raw_images, images_dir)

    data = json.loads(rendered.model_dump_json(exclude={"images"}))
    markdown = data.get("markdown", "")
    metadata = data.get("metadata", {})

    if not markdown.strip():
        print("[ERROR] No content extracted from PDF.")
        return

    print(f"[3/3] Parsing sections {sorted(allowed_types)}...")
    sections = parse_markdown(markdown, allowed_types, saved_images)

    # Summary counts
    counts = {"text": 0, "equation": 0, "table": 0, "figure": 0}
    for s in sections:
        if s.get("text"):
            counts["text"] += 1
        counts["equation"] += len(s.get("equations", []))
        counts["table"]    += len(s.get("tables", []))
        counts["figure"]   += len(s.get("figures", []))

    output = {
        "source": Path(pdf_path).name,
        "metadata": {
            "pages": len(metadata.get("page_stats", [])),
            "toc": [
                {"title": e["title"], "page": e["page_id"]}
                for e in metadata.get("table_of_contents", [])
            ],
        },
        "sections": sections,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved to '{output_path}'")
    print(f"  Pages    : {output['metadata']['pages']}")
    print(f"  Sections : {len(sections)}")
    print(f"  text     : {counts['text']} section(s) with text")
    print(f"  equations: {counts['equation']}")
    print(f"  tables   : {counts['table']}")
    print(f"  figures  : {counts['figure']} (images in '{images_dir}')")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pdf_parser",
        description=(
            "Extract content from a PDF into structured JSON grouped by section. "
            "Text is always included. Use flags to add more content types."
        ),
    )
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument(
        "-o", "--output",
        default="output.json",
        help="Path to the output JSON file (default: output.json)",
    )
    parser.add_argument("--tables",    action="store_true", help="Include tables")
    parser.add_argument("--figures",   action="store_true", help="Include figures and captions")
    parser.add_argument("--equations", action="store_true", help="Include equations")

    args = parser.parse_args()

    allowed_types = {"text"}
    if args.tables:    allowed_types.add("table")
    if args.figures:   allowed_types.update({"figure", "caption"})
    if args.equations: allowed_types.add("equation")

    process(args.input, args.output, allowed_types)