# Data Directory — RAG Source Documents

This directory holds the PDF documents ingested by the RAG Engine to provide
engineering context to the Validator Agent during design evaluation.

## Expected File Types

- **PDF (.pdf)** — The primary supported format for RAG ingestion.
  Both text-searchable PDFs and image-heavy PDFs are supported.

## Multimodal RAG

The RAG Engine supports **multimodal ingestion**:

- **Text content** — Extracted from PDF pages and split into searchable chunks.
- **Images/diagrams** — Extracted from PDF pages using PyMuPDF, then captioned
  by `meta/llama-3.2-90b-vision-instruct` (NVIDIA NIM vision model). The
  generated captions are stored as additional text chunks in the vector store.

This means engineering diagrams, dimension drawings, stress plots, and
technical illustrations in your PDFs will be understood and used during
validation — not just the text.

## Naming Conventions

Use descriptive, lowercase filenames with hyphens separating words:

```text
<category>-<topic>-<optional-version>.pdf
```

### Categories

| Prefix            | Description                               |
|-------------------|-------------------------------------------|
| `standards-`      | Engineering standards (ISO, ASTM, etc.)   |
| `materials-`      | Material datasheets and property tables   |
| `design-`         | Design guidelines and best practices      |
| `manufacturing-`  | Manufacturing constraints (FDM, CNC, etc.)|
| `aerospace-`      | Aerospace/drone-specific references       |

### Examples

```text
standards-iso-2768-tolerances.pdf
materials-pla-datasheet-v2.pdf
design-quadcopter-arm-geometry.pdf
manufacturing-fdm-minimum-thickness.pdf
aerospace-drone-structural-loads.pdf
```

## Vector Store Persistence

After the first successful ingestion, the RAG Engine persists the FAISS index
to `data/vectorstore/`. On subsequent startups, the engine loads the persisted
index instead of re-processing PDFs.

To force re-ingestion (e.g., after adding new PDFs), delete the `data/vectorstore/` directory:

```bash
rmdir /s /q data\vectorstore
```

## Notes

- Text documents are split into 500-character chunks with 50-character overlap.
- Images smaller than 100×100 pixels are skipped (icons, logos, decorations).
- Embeddings use NVIDIA Nemotron Embed (NV-Embed-QA) via NVIDIAEmbeddings.
- Image captioning uses meta/llama-3.2-90b-vision-instruct via ChatNVIDIA.
- Unparseable PDFs are skipped with a warning — they do not block ingestion.
- If no PDFs are present, the system operates in degraded mode (empty context).
- First ingestion with images may take a few minutes (one vision API call per image).
