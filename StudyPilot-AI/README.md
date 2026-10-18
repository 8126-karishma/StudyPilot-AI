# StudyPilot AI - Smart Notes to Learning Assistant

StudyPilot AI turns your own PDF or TXT study notes into a small learning assistant: summaries, questions answered **from your notes with page references**, practice quizzes, and key concepts. It runs locally in a Streamlit web app, and the core workflow needs no paid API and no large model.

> Built as a beginner-friendly but professionally structured AI/ML portfolio project.

## 1. Problem statement

Students collect long notes and PDFs but have little time to re-read them. Generic chatbots can answer questions, but they may invent facts and cannot show which page an answer came from. StudyPilot AI answers only from the uploaded document and shows its evidence.

## 2. Objectives

- Load PDF/TXT notes safely and keep page numbers.
- Summarise long documents without exceeding model limits.
- Answer questions with retrieval-augmented generation (RAG) and visible sources.
- Generate and score multiple-choice quizzes from the notes.
- Extract key concepts with definitions.
- Work on an ordinary CPU, degrade gracefully, and never pretend a model was used when it was not.

## 3. Features

| Feature | How it works | Needs a model download? |
|---|---|---|
| Upload | PDF (page by page) and TXT, type/size/page/text limits, empty and scanned-PDF detection | No |
| Summary | Extractive short + detailed summaries, processed block by block; optional local-model short summary | No (model optional) |
| Ask My Notes | Chunks -> embeddings -> cosine similarity -> evidence -> answer, with source pages and a "no relevant passage" state | No (neural embeddings optional) |
| Semantic search | Browse the best-matching passages with similarity scores | No |
| Quiz | Definition and fill-in-the-blank questions, validation, de-duplication, hidden answers until submit, scoring and feedback | No (model optional) |
| Key concepts | Frequency + definition detection, with supporting sentence and page | No |
| Demo mode | Bundled original notes (`data/sample_notes.txt`, plus a generated `sample_notes.pdf`) | No |

Honest note on "AI": by default the app uses classical NLP (TF-IDF, cosine similarity, rules). Neural sentence embeddings (Sentence Transformers) and a local language model (FLAN-T5 via Transformers) are supported as **optional** upgrades. The interface always labels which method produced a result.

## 4. Technology stack

Python 3.10+, Streamlit, PyMuPDF, scikit-learn, NumPy, python-dotenv, Sentence Transformers (optional neural embeddings), Hugging Face Transformers + PyTorch (optional generation), Pytest, Git/GitHub.

## 5. Architecture and workflow

```
Upload (PDF/TXT) -> validate -> extract per page -> clean text
      -> chunk with overlap (keeps page number)
      -> embed chunks (neural, or TF-IDF fallback)
      -> question -> embed -> cosine similarity -> top passages
      -> answer (extractive, or optional local model) + source pages
```

Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Concepts in plain language

- **Chunking** - cutting a long document into small overlapping pieces so each piece can be searched on its own. Overlap keeps sentences that fall on a boundary.
- **Embedding** - a list of numbers representing the meaning of a text; similar meanings give similar numbers.
- **Cosine similarity** - a score from 0 to 1 for how closely two embeddings point in the same direction. Higher means more related.
- **RAG (Retrieval-Augmented Generation)** - first *retrieve* the most relevant passages, then *generate* (or extract) an answer from them instead of from the model's memory.
- **Source grounding** - every answer is tied to retrieved passages and page numbers, and the app says so when nothing relevant is found.

## 6. Project structure

```
StudyPilot-AI/
├── app.py                      Streamlit interface
├── requirements.txt            runtime dependencies
├── requirements-dev.txt        runtime + pytest
├── pyproject.toml              package metadata + pytest config
├── .env.example                optional settings (no secrets)
├── .streamlit/config.toml      theme and upload limit
├── src/studypilot/
│   ├── config.py               settings and limits
│   ├── pdf_reader.py           safe PDF/TXT loading and validation
│   ├── text_processor.py       cleaning, sentences, chunking, data classes
│   ├── embeddings.py           neural + TF-IDF embedders, fallback logic
│   ├── retriever.py            cosine-similarity search
│   ├── generation.py           optional local Hugging Face generator
│   ├── summarizer.py           extractive + optional model summaries
│   ├── rag_pipeline.py         retrieval + grounded answers
│   ├── quiz_generator.py       questions, validation, scoring
│   └── study_utils.py          definitions, key concepts, helpers
├── data/                       sample_notes.txt, sample_notes.pdf
├── tests/                      pytest suite (103 tests)
├── docs/                       INSTALLATION, ARCHITECTURE, GITHUB_GUIDE
├── scripts/                    run_tests.py, make_sample_pdf.py
└── assets/screenshots/         (empty - see section 11)
```

`generation.py` is an addition to the suggested layout: it isolates the optional model so the summariser, RAG and quiz modules share it without circular imports.

## 7. Installation and execution

Full instructions: [docs/INSTALLATION.md](docs/INSTALLATION.md).

**Windows (PowerShell)**

```powershell
cd StudyPilot-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**Linux / macOS**

```bash
cd StudyPilot-AI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the printed local URL (normally http://localhost:8501) and press **Try demo notes**.

**First run:** with `sentence-transformers` installed, the first search downloads the `all-MiniLM-L6-v2` model (roughly 90 MB) into your Hugging Face cache; later runs reuse it. If the download is impossible (offline, blocked), the app shows a notice and uses TF-IDF instead. `pip install -r requirements.txt` also installs PyTorch, which is a large download; if you only want the lightweight mode, remove the `sentence-transformers` line from `requirements.txt`.

**Optional local language model:** `pip install transformers torch`, then tick "Use optional local language model" in the sidebar (downloads FLAN-T5 small on first use).

## 8. Configuration

Copy `.env.example` to `.env` and edit values (all optional). Variables use the `STUDYPILOT_` prefix: file/page/text limits, chunk size and overlap, `TOP_K`, the embedding and generator model names, and switches for neural embeddings and generation.

## 9. Sample usage

1. Open **Upload** and choose a PDF/TXT, or press **Try demo notes**.
2. **Summary** -> *Generate summary* for a short and a detailed version.
3. **Ask My Notes** -> try "What is overfitting?", "How does gradient descent work?" or "What is cosine similarity?". Ask something unrelated ("Who won the 1998 World Cup?") to see the "no relevant passage" behaviour.
4. **Quiz** -> *Generate quiz*, answer, *Submit answers* to see score, percentage and explanations.
5. **Key Concepts** -> expand a concept to see its definition and page/section.

TXT files have no pages, so they are split into numbered **sections**; PDFs report real **page** numbers.

## 10. Testing

```bash
python scripts/run_tests.py        # or: pytest
```

**Actual result in the build environment:** `103 passed` in about 5 seconds (Python 3.12.3, Linux, Streamlit 1.66.0, PyMuPDF 1.28.2, scikit-learn 1.8.0, NumPy 2.4.4, pytest 9.1.1). In that environment `sentence-transformers`, `transformers` and `torch` were **not installed**. Tests cover PDF extraction, empty/scanned/corrupt/encrypted PDFs, type and size validation, normalisation, chunking and overlap, retrieval and page metadata, grounded answers, the no-evidence case, quiz validation/scoring/missing answers, malformed model output, fallbacks when optional models are unavailable, and headless UI smoke tests (Streamlit `AppTest`). Model inference is mocked, so tests need no downloads.

Also verified: `streamlit run app.py` started a server whose health endpoint returned HTTP 200.

## 11. Screenshots

None are included: the environment this was built in had no browser, so no real screenshots could be captured, and none are faked. To add yours, run the app, save images in `assets/screenshots/`, and reference them here.

## 12. Known limitations

- **Not tested here:** the real Sentence Transformers model, the real FLAN-T5 model, Windows, macOS, and Python 3.10/3.11. Their code paths are covered only by mocks/fallback tests. Please run the app once with them on your machine.
- Extractive answers copy sentences from the notes; they are accurate to the text but not fluent prose. TF-IDF matches words, not meaning, so paraphrased questions may find nothing.
- The similarity thresholds (`0.10` TF-IDF, `0.30` neural) are reasonable heuristics, not tuned on a benchmark.
- FLAN-T5 *small* is weak at summaries, answers and JSON-style quiz output. Quiz questions it writes are discarded unless they parse and are supported by the source passage; the rule-based generator fills the rest.
- Quiz and concept quality depends on how clearly the notes state definitions ("X is a ..."). Very short or unstructured notes may give few questions.
- No OCR: scanned PDFs are rejected with an explanation. English text only. One document at a time. Tables, formulas and images in PDFs are not interpreted.
- No accuracy figures are claimed; nothing here has been evaluated against a benchmark.

## 13. Privacy and security notes

- Uploads are read in memory, never executed, and not written to disk by the app. Nothing is sent to external services by default.
- The only network activity is Hugging Face model downloads when you enable neural embeddings or the local language model (the documents themselves are never uploaded).
- Notes are treated as untrusted data: file type/size/page/text limits, PDF signature checks, filename sanitising, Markdown escaping when text is displayed, and prompts that mark the notes as data. This reduces but cannot eliminate prompt-injection risk when a language model is enabled.
- The UI shows friendly errors only; details go to the log.
- Keep `.env` out of Git (it is in `.gitignore`).

## 14. Future improvements

OCR for scanned PDFs (e.g. Tesseract), multi-document collections, persistent vector index, evaluation set for retrieval quality, multilingual models, flashcard export, optional hosted-LLM integration (with a clear privacy notice).

## 15. Contribution guidelines

Fork, create a branch, add tests for changes, run `python scripts/run_tests.py`, and open a pull request describing the change. Keep new dependencies lightweight and never commit secrets or private documents.

## 16. License

MIT - see [LICENSE](LICENSE). Edit the copyright line to your full name before publishing.

## 17. GitHub setup

See [docs/GITHUB_GUIDE.md](docs/GITHUB_GUIDE.md). Quick version:

```bash
git init
git add .
git commit -m "Initial commit: StudyPilot AI"
git branch -M main
git remote add origin https://github.com/<your-username>/StudyPilot-AI.git
git push -u origin main
```

## 18. What I learned (interview talking points)

Write these in your own words after you run the project. Starting points:

- **RAG end to end:** why retrieving evidence first reduces invented answers, and how chunk size and overlap change what can be found.
- **Embeddings vs TF-IDF:** meaning-based vs keyword-based similarity, and why a fallback keeps the app usable offline.
- **Cosine similarity and thresholds:** why "no relevant result" is a valid output and how a threshold decides it.
- **Engineering for failure:** optional dependencies, graceful degradation, input validation, untrusted-input handling.
- **Honest evaluation:** what the tests prove (behaviour with mocks) versus what they do not (model quality), and how you would measure retrieval quality next (e.g. a labelled question set, recall@k).
- **Trade-offs:** extractive vs generative answers, small CPU models vs quality.
