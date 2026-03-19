# PDF_Parser

A Python CLI tool that converts PDF files into structured JSON, grouped by section. Built on [marker-pdf](https://github.com/VikParuchuri/marker) for accurate layout detection across all PDF types — clean reports, scanned documents, multi-column academic papers, and books.

Designed as the extraction step of a RAG / vector database pipeline.

---

## Output Format

```json
{
  "source": "paper.pdf",
  "metadata": {
    "pages": 15,
    "toc": [
      { "title": "1 Introduction", "page": 1 },
      { "title": "3.2 Attention",  "page": 4 }
    ]
  },
  "sections": [
    {
      "title": "3.2 Attention",
      "text": "An attention function can be described as mapping a query...",
      "equations": [
        "$$Attention(Q, K, V) = softmax(\\frac{QK^{T}}{\\sqrt{d_k}})V$$"
      ],
      "tables": [
        {
          "caption": "Table 1: Maximum path lengths per layer type.",
          "content": "| Layer Type | Complexity |\n|------------|------------|\n| Self-Attention | O(n^2) |"
        }
      ],
      "figures": [
        {
          "caption": "Figure 1: The Transformer model architecture.",
          "image_id": "_page_2_Figure_0.jpeg",
          "image_path": "output/figures/_page_2_Figure_0.jpeg.png"
        }
      ]
    }
  ]
}
```

### Section fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Section heading |
| `text` | string | All paragraph text merged into one string |
| `equations` | list | LaTeX strings |
| `tables` | list | `{caption, content}` — content is raw Markdown table |
| `figures` | list | `{caption, image_id, image_path}` — image saved as PNG on disk |

See [`example_output.json`](example_output.json) for a full example.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/your-username/PDF_Parser.git
cd PDF_Parser
pip install -r requirements.txt
```

> **Note:** marker-pdf will download ML model weights (~3GB) on the first run.

---

## Usage

```bash
python pdf_parser.py <input.pdf> [options]
```

### Options

| Flag | Description |
|---|---|
| `-o, --output` | Output file path (default: `output.json`) |
| `--tables` | Include tables |
| `--figures` | Include figures, captions, and saved images |
| `--equations` | Include equations |

**Text is always extracted.** Everything else is opt-in.

### Examples

```bash
# Text only (default)
python pdf_parser.py report.pdf

# Custom output path
python pdf_parser.py report.pdf -o results.json

# Text + tables + figures
python pdf_parser.py report.pdf --tables --figures

# Extract everything
python pdf_parser.py report.pdf --tables --figures --equations -o full.json

# Help
python pdf_parser.py --help
```

### Example run

```
[1/3] Loading marker-pdf models...
[2/3] Converting 'paper.pdf'...
      Saving 6 figure image(s)...
[3/3] Parsing sections ['equation', 'figure', 'table', 'text']...

Done! Saved to 'full.json'
  Pages    : 15
  Sections : 27
  text     : 27 section(s) with text
  equations: 8
  tables   : 4
  figures  : 6 (images in 'full/figures')
```

---

## Figure Images

When `--figures` is passed, extracted images are saved as PNGs next to the output JSON:

```
full.json
full/
  figures/
    _page_2_Figure_0.jpeg.png
    _page_3_Figure_1.jpeg.png
    ...
```

Each figure block in the JSON links back to its saved image via `image_id` and `image_path`, making it straightforward to load the image for multimodal embedding later.

---

## Why JSON first?

The parser deliberately stops at JSON — no chunking, no embedding. This keeps the slow extraction step (marker-pdf runs ~10–15s per page on CPU) separate from the fast, tuneable chunking step.

```
PDF  ──►  marker-pdf  ──►  JSON            (this tool)
                             │
                             ▼
                    chunk ──► embed ──► vector DB
```

Storing clean per-section JSON means you can experiment with chunk sizes, overlap, and embedding models without re-parsing the PDF. For multimodal RAG, text fields embed with `text-embedding-3` and figure images embed with CLIP — both go into the same vector DB.

---

## Performance

marker-pdf uses ML models for layout detection. On CPU expect roughly **10–15 seconds per page**. GPU significantly speeds this up but requires CUDA compute capability `sm_70+` and at least 6GB VRAM.

To disable the CPU override (if you have a compatible GPU), remove this line from `pdf_parser.py`:

```python
os.environ["TORCH_DEVICE"] = "cpu"
```

---

## Notes

- Not all equations will be detected as LaTeX — complex inline math may appear as plain text
- Tables with captions embedded inside `|` rows may not have their caption extracted
- Figures are represented by captions and saved PNGs — vector image data is not preserved
- Headers, footers, and table of contents entries are automatically stripped by marker-pdf

---

## Roadmap

- [ ] Batch processing (multiple PDFs at once)
- [ ] Per-page JSON output option
- [ ] Optional chunking step for direct vector DB ingestion
- [ ] Multimodal embedding integration (CLIP for figures)