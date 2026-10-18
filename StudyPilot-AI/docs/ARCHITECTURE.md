# Architecture

The UI (`app.py`) only handles layout and session state. All logic lives in `src/studypilot/` so it can be tested without Streamlit.

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` dataclass, environment overrides (`STUDYPILOT_*`), logging setup |
| `text_processor.py` | `Document`/`PageText`/`Chunk`, `normalize_text`, `split_sentences`, `chunk_text`, `chunk_pages` |
| `pdf_reader.py` | Validation (type, size, PDF signature), PyMuPDF extraction per page, TXT decoding, typed errors with safe messages |
| `embeddings.py` | `TfidfEmbedder`, `SentenceTransformerEmbedder` (lazy load, per-text cache), `get_embedder` with fallback note |
| `retriever.py` | Builds the index once, cosine-similarity search with a relevance threshold |
| `generation.py` | Optional local seq2seq generator (lazy, shared); raises `GenerationUnavailableError` |
| `summarizer.py` | Block-wise extractive summarisation; optional model summary with fallback |
| `rag_pipeline.py` | Chunk + index a document; `ask()` returns an `Answer` with method label, sources, notes |
| `quiz_generator.py` | Rule-based and model-based questions, validation, de-duplication, scoring |
| `study_utils.py` | Definition detection, key concepts, Markdown escaping, formatting helpers |

## Data flow

1. `load_document` -> `Document(pages=[PageText(number, text)])`. PDF pages keep their real numbers (blank pages are skipped but numbering is preserved). TXT is split into numbered sections.
2. `chunk_pages` -> `Chunk(page, label, text)`; chunks never cross pages, so page metadata is always correct.
3. `Retriever` fits the embedder and stores the chunk matrix; `search` returns chunks above the threshold.
4. `RAGPipeline.ask`: no results -> "no relevant passage" (no model call). Otherwise, with a generator, the retrieved text is placed in a prompt marked as untrusted data; without one (or on failure) the best sentences are copied from the chunks, with their page labels.
5. Quiz questions carry `source_text` and `source_page`; model-written questions must parse as JSON, validate, and have an answer supported by the passage.

## Design decisions

- **Fallbacks everywhere:** every optional model has a deterministic non-model path, and the UI labels which one ran.
- **No answer without evidence:** thresholds make "not found" a first-class result.
- **Caching:** neural models are loaded once per process; vectors are cached per text; the search index and summaries are cached in session state keyed by the document fingerprint.
- **Safety:** uploads stay in memory; text shown in the UI is Markdown-escaped; user-facing errors never include paths or stack traces.
