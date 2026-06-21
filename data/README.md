# Data Directory — RAG Source Documents

This directory holds the PDF documents ingested by the RAG Engine to provide
engineering context to the Validator Agent during design evaluation.

## Expected File Types

- **PDF (.pdf)** — The only supported format for RAG ingestion.
  All documents must be text-searchable PDFs (not scanned images).

## Naming Conventions

Use descriptive, lowercase filenames with hyphens separating words:

```
<category>-<topic>-<optional-version>.pdf
```

### Categories

| Prefix | Description |
|--------|-------------|
| `standards-` | Engineering standards (ISO, ASTM, etc.) |
| `materials-` | Material datasheets and property tables |
| `design-` | Design guidelines and best practices |
| `manufacturing-` | Manufacturing constraints (FDM, CNC, etc.) |
| `aerospace-` | Aerospace/drone-specific references |

### Examples

```
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

To force re-ingestion, delete the `data/vectorstore/` directory.

## Notes

- Documents are split into 500-character chunks with 50-character overlap.
- Embeddings use NVIDIA Nemotron Embed (NV-Embed-QA) via NVIDIAEmbeddings.
- Unparseable PDFs are skipped with a warning — they do not block ingestion.
- If no PDFs are present, the system operates in degraded mode (empty context).
