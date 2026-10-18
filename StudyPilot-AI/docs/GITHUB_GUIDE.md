# GitHub guide

## 1. Prepare

```bash
cd StudyPilot-AI
python scripts/run_tests.py          # make sure everything passes
```

Edit `LICENSE` (your full name) and the README (add screenshots you captured yourself).

## 2. Create the repository

On https://github.com/new create an empty repository named `StudyPilot-AI` (no README/licence/.gitignore, because the project already has them).

## 3. Push

```bash
git init
git add .
git status                             # confirm no .env, .venv or model files are listed
git commit -m "Initial commit: StudyPilot AI"
git branch -M main
git remote add origin https://github.com/<your-username>/StudyPilot-AI.git
git push -u origin main
```

If asked to authenticate, use a GitHub personal access token or the GitHub CLI (`gh auth login`) instead of a password.

## 4. Suggested repository settings

Description: "Upload notes -> summaries, grounded Q&A, quizzes and key concepts (Streamlit, RAG)". Topics: `streamlit`, `rag`, `nlp`, `sentence-transformers`, `education`.

## Optional: deploy on Streamlit Community Cloud

These steps have **not** been carried out or verified for this project.

1. Push the repository to GitHub (public, or grant access).
2. Go to https://share.streamlit.io, choose *New app*, select the repository, branch `main`, main file `app.py`, and a Python version (3.11 or 3.12 recommended).
3. The free tier has limited memory, so deploy in lightweight mode (remove `sentence-transformers` from `requirements.txt`) or expect the neural model to be slow or fail; the app falls back to TF-IDF.
4. Privacy: on a hosted app, uploaded notes are processed on the hosting provider's servers. Do not upload private material, and tell users so.
