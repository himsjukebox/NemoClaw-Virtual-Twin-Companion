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

## Recommended Reference PDFs

For the best RAG grounding, download one or more of these open-access papers
and place them in this directory:

### Primary Recommendation

**"Design and Development of Unibody Quadcopter Structure Using Optimization
and Additive Manufacturing Techniques"**
- Source: MDPI Designs journal (Open Access, CC BY 4.0)
- URL: https://www.mdpi.com/2411-9660/6/1/8
- Download: Click "Download PDF" on the page
- Save as: `data/design-unibody-quadcopter-optimization-mdpi-2022.pdf`
- Why: Covers structural FEA, PLA/ABS material selection, arm geometry
  optimization, thrust-to-weight analysis, and 3D printing constraints.
  Directly relevant to every metric the physics engine computes.

### Additional Recommendations

1. **"Design and Analysis of 3D Printed Quadrotor Frame"**
   - URL: https://www.researchgate.net/publication/331813111
   - Save as: `data/design-3d-printed-quadrotor-frame-2019.pdf`
   - Why: ABS-PC and carbon fiberglass FDM analysis, structural simulation
     under lift/drag/thrust, safety factor calculations.

2. **"Quadcopter Design, Construction and Testing"**
   - URL: https://www.researchgate.net/publication/331859602
   - Save as: `data/design-quadcopter-construction-testing-2019.pdf`
   - Why: Weight estimation methodology, component selection based on payload,
     motor-prop matching, and flight time calculations.

### After Adding PDFs

Delete the old vector store to force re-ingestion:

```bash
rmdir /s /q data\vectorstore     # Windows
# rm -rf data/vectorstore        # Linux/macOS
```

Then restart the app — the RAG Engine will re-ingest on the next validator run.
