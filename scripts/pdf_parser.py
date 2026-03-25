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
import torch
import argparse
from pathlib import Path

os.environ["TORCH_DEVICE"] = "cuda"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    lines = markdown.split("\n")
    sections = []

    current = {
        "title": "Preamble",
        "text_lines": [],
        "equations": [],
        "tables": [],
        "figures": [],
        "pending_table_caption": None,
        "last_text_line": None,  # ← IMPORTANT (for captions above images)
    }

    # ── Skip patterns ────────────────────────────────────────────────────────
    SKIP_LINE_PATTERNS = [
        re.compile(r'STANDARDNI OPERATIVNI POSTOPEK', re.IGNORECASE),
        re.compile(r'^Velja od:', re.IGNORECASE),
        re.compile(r'^Velja do:', re.IGNORECASE),
        re.compile(r'ZAUPNO', re.IGNORECASE),
        re.compile(r'^Oznaka:', re.IGNORECASE),
        re.compile(r'NEKONTROLIRANA KOPIJA', re.IGNORECASE),
        re.compile(r'NAVODILO ZA INTEGRACIJO KROMATOGRAFSKIH VRHOV', re.IGNORECASE),
    ]

    SKIP_HEADING_PATTERNS = [
        re.compile(r'KAZALO', re.IGNORECASE),
    ]

    SKIP_TABLE_PATTERNS = [
        re.compile(r'Pripravila', re.IGNORECASE),
        re.compile(r'Pregledala', re.IGNORECASE),
        re.compile(r'STANDARDNI OPERATIVNI POSTOPEK', re.IGNORECASE),
    ]

    # ── Helpers ──────────────────────────────────────────────────────────────
    def is_heading(line):
        return bool(re.match(r'^#{1,6}\s+\S', line))

    def clean_heading(line):
        return re.sub(r'^#{1,6}\s+', '', line).strip()

    def is_skip_line(line):
        return any(p.search(line) for p in SKIP_LINE_PATTERNS)

    def is_noise(text):
        text = text.strip()
        if len(text) == 1 and text.isalpha():
            return False
        return bool(re.fullmatch(r'[\d\s\-–—|._/\\]+', text)) or len(text) < 3

    def clean_text(line):
        line = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', line)
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
        return line.strip()

    def is_skip_table(table_lines):
        data_rows = [l for l in table_lines if l.startswith('|') and '---' not in l][:2]
        all_cells = ' '.join(data_rows)
        return any(p.search(all_cells) for p in SKIP_TABLE_PATTERNS)

    def flush_section():
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
        current["pending_table_caption"] = None
        current["last_text_line"] = None

    # ── Main loop ────────────────────────────────────────────────────────────
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if is_skip_line(line):
            i += 1
            continue

        stripped = re.sub(r'^#{1,6}\s+', '', line).strip()
        if any(p.search(stripped) for p in SKIP_HEADING_PATTERNS):
            i += 1
            continue

        # ── Heading ─────────────────────────────────────────────────────────
        if is_heading(line):
            flush_section()
            current["title"] = clean_heading(line)
            i += 1
            continue

        # ── Image (caption ABOVE) ───────────────────────────────────────────
        img_match = re.match(r'!\[.*?\]\((.+?)\)', line)
        if img_match:
            img_id = img_match.group(1)

            caption = current.get("last_text_line")

            # ── RULES ─────────────────────────────────────────────
            is_picture = "Picture_" in img_id
            has_caption = caption and not is_noise(caption)

            # keep image only if:
            # - NOT a Picture OR
            # - it has a meaningful caption
            keep_image = (not is_picture) or has_caption

            if keep_image and "figure" in allowed_types:
                current["figures"].append({
                    "caption": caption if has_caption else "",
                    "image_id": img_id,
                    "image_path": saved_images.get(img_id, ""),
                })

            # reset caption buffer
            current["last_text_line"] = None

            i += 1
            continue
        # ── Table caption ───────────────────────────────────────────────────
        clean_line = clean_text(line)
        if re.match(r"^Table\s*\d+", clean_line, re.IGNORECASE):
            current["pending_table_caption"] = clean_line
            i += 1
            continue

        # ── Equation ────────────────────────────────────────────────────────
        if line.strip().startswith("$$"):
            if line.strip().endswith("$$") and len(line.strip()) > 4:
                if "equation" in allowed_types:
                    current["equations"].append(line.strip())
                i += 1
                continue

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

        # ── Table ───────────────────────────────────────────────────────────
        if line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1

            if not is_skip_table(table_lines) and "table" in allowed_types:
                current["tables"].append({
                    "caption": current["pending_table_caption"] or "",
                    "content": "\n".join(table_lines),
                })

            current["pending_table_caption"] = None
            continue

        # ── Text ────────────────────────────────────────────────────────────
        if "text" in allowed_types:
            clean = clean_text(line)
            if clean and not is_noise(clean):
                current["text_lines"].append(clean)
                current["last_text_line"] = clean  # ← store for figure caption

        i += 1

    flush_section()
    return sections

def build_toc_tree(sections):
    toc = []

    for s in sections:
        title = s["title"]
        match = re.match(r'^(\d+(\.\d+)*)\s+(.*)', title)

        if match:
            level = match.group(1).count('.') + 1
            toc.append({
                "level": level,
                "number": match.group(1),
                "title": match.group(3),
            })
        else:
            toc.append({
                "level": 1,
                "number": None,
                "title": title,
            })

    return toc

def build_toc_from_sections(sections):
    toc = []
    for s in sections:
        # Match leading number, optional dot or dash, then whitespace
        match = re.match(r'^(\d+(\.\d+)*)(?:[.\-])?\s*(.*)', s['title'])
        if match:
            number = match.group(1)
            title  = match.group(3).strip()
        else:
            number = None
            title  = s['title'].strip()
        toc.append({"number": number, "title": title})
    return toc

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
            "toc": build_toc_from_sections(sections),
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