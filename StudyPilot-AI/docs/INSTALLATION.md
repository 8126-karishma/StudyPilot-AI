# Installation

## Requirements

- Python 3.10 or newer (check with `python --version`)
- About 1 GB free disk space for the lightweight setup; several GB if PyTorch and models are installed
- Internet only for installing packages and (optionally) downloading models

## Tested versions

The project was tested on Python 3.12.3 (Linux) with: Streamlit 1.66.0, PyMuPDF 1.28.2, scikit-learn 1.8.0, NumPy 2.4.4, python-dotenv 1.2.2, pytest 9.1.1. `requirements.txt` uses lower bounds so pip can pick versions compatible with your Python. Sentence Transformers, Transformers and PyTorch were **not** installed during testing.

## Windows (PowerShell)

```powershell
# 1. Extract StudyPilot-AI-GitHub-Ready.zip (right-click -> Extract All), then:
cd path\to\StudyPilot-AI
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For Command Prompt use `.venv\Scripts\activate.bat`.

## Linux / macOS

```bash
unzip StudyPilot-AI-GitHub-Ready.zip && cd StudyPilot-AI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Lightweight mode (no PyTorch)

Delete the `sentence-transformers` line from `requirements.txt` before installing (or install packages one by one). The app then uses TF-IDF keyword matching and shows a notice on the Ask page.

## Running the tests

```bash
pip install -r requirements-dev.txt
python scripts/run_tests.py
```

## First run and model downloads

- Neural embeddings: `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) is downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`.
- Optional language model: `pip install transformers torch`, tick the sidebar option; `google/flan-t5-small` is downloaded on first use. Expect slower responses on CPU.
- ZIP/Git never contain model weights.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the virtual environment, or run `python -m streamlit run app.py` |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process Bypass`, then activate again |
| `pip install` of PyTorch is very slow/large | Use lightweight mode, or install CPU wheels: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Notice says neural embeddings are unavailable | The package or model could not be loaded (offline, firewall, no disk space). The app keeps working with TF-IDF; fix the network/package and rerun |
| `No selectable text was found` | The PDF is scanned images; OCR is not supported. Use a text-based PDF or TXT |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Import error for `studypilot` | Run commands from the project root, or `pip install -e .` |
| Slow first question | Models are loading; later questions reuse them |
