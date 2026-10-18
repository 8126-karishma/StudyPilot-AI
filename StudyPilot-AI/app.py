"""StudyPilot AI - Streamlit user interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:  # lets `streamlit run app.py` work without installing the package
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from studypilot import __version__  # noqa: E402
from studypilot.config import SAMPLE_NOTES_PATH, Settings, configure_logging  # noqa: E402
from studypilot.embeddings import get_embedder  # noqa: E402
from studypilot.generation import get_generator  # noqa: E402
from studypilot.pdf_reader import DocumentError, load_document  # noqa: E402
from studypilot.quiz_generator import generate_quiz, score_quiz  # noqa: E402
from studypilot.rag_pipeline import RAGPipeline  # noqa: E402
from studypilot.study_utils import (  # noqa: E402
    escape_markdown,
    estimate_reading_minutes,
    extract_key_concepts,
    format_file_size,
)
from studypilot.summarizer import summarize  # noqa: E402
from studypilot.text_processor import Document  # noqa: E402

logger = logging.getLogger("studypilot.app")

PAGES = ["Dashboard", "Upload", "Summary", "Ask My Notes", "Quiz", "Key Concepts"]

CSS = """
<style>
.sp-logo {display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem;}
.sp-logo-mark {width:2.4rem; height:2.4rem; border-radius:.7rem; display:flex; align-items:center;
  justify-content:center; font-size:1.4rem; color:#fff; background:linear-gradient(135deg,#4F46E5,#14B8A6);}
.sp-logo-text {font-weight:700; font-size:1.25rem; line-height:1.1;}
.sp-logo-sub {font-size:.75rem; opacity:.7;}
.sp-hero {padding:1.4rem 1.6rem; border-radius:1rem; color:#fff; margin-bottom:1rem;
  background:linear-gradient(135deg,#4F46E5 0%,#6366F1 55%,#14B8A6 100%);}
.sp-hero h1 {margin:0 0 .3rem 0; font-size:1.9rem; color:#fff;}
.sp-hero p {margin:0; font-size:1.02rem; opacity:.95;}
.sp-card {border:1px solid rgba(128,128,128,.28); border-radius:.9rem; padding:1rem 1.1rem; height:100%;}
.sp-card h4 {margin:0 0 .35rem 0; font-size:1.02rem;}
.sp-card p {margin:0; font-size:.92rem; opacity:.85;}
@media (max-width: 640px) {.sp-hero h1 {font-size:1.45rem;} .sp-hero {padding:1rem;}}
</style>
"""


# ----------------------------------------------------------------- state helpers
def get_settings() -> Settings:
    return Settings.from_env()


def init_state(settings: Settings) -> None:
    defaults = {
        "nav": "Dashboard", "doc": None, "pipeline": None, "pipeline_key": None, "embedder_note": None,
        "chat": [], "summaries": {}, "quiz": None, "quiz_counter": 0,
        "use_dense": settings.use_dense_embeddings, "use_generative": settings.use_generative_models,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(page: str) -> None:
    st.session_state["nav"] = page


def set_document(doc: Document) -> None:
    st.session_state.update(
        doc=doc, pipeline=None, pipeline_key=None, embedder_note=None, chat=[], summaries={}, quiz=None
    )


def clear_session() -> None:
    for key in ("doc", "pipeline", "pipeline_key", "embedder_note", "quiz"):
        st.session_state[key] = None
    st.session_state["chat"] = []
    st.session_state["summaries"] = {}


def current_generator(settings: Settings):
    """The optional local model (loaded lazily on first use) or ``None``."""
    return get_generator(settings) if st.session_state.get("use_generative") else None


def ensure_pipeline(settings: Settings) -> RAGPipeline | None:
    """Build (or reuse) the search index for the loaded document."""
    doc: Document | None = st.session_state.doc
    if doc is None:
        return None
    key = (doc.fingerprint, bool(st.session_state.use_dense))
    if st.session_state.pipeline_key != key:
        with st.spinner("Preparing your notes for search... (the first run may take a minute while models load)"):
            try:
                embedder, note = get_embedder(settings, prefer_dense=st.session_state.use_dense)
                st.session_state.pipeline = RAGPipeline(doc, embedder, settings)
            except Exception as exc:
                logger.exception("Could not build the search index")
                st.error("Something went wrong while preparing the notes for search. Please try another file.")
                st.session_state.pipeline = None
                st.session_state.pipeline_key = None
                logger.debug("%s", exc)
                return None
        st.session_state.pipeline_key = key
        st.session_state.embedder_note = note
    return st.session_state.pipeline


def require_document() -> Document | None:
    doc: Document | None = st.session_state.doc
    if doc is None:
        st.info("No notes loaded yet. Upload a PDF or TXT file, or try the bundled demo notes.")
        c1, c2 = st.columns(2)
        c1.button("Go to Upload", key="empty_upload", on_click=goto, args=("Upload",))
        c2.button("Load demo notes", key="empty_demo", on_click=load_demo)
    return doc


def load_demo() -> None:
    try:
        data = SAMPLE_NOTES_PATH.read_bytes()
        set_document(load_document("sample_notes.txt", data))
        st.session_state["flash"] = "Demo notes loaded - explore the other pages from the sidebar."
    except (OSError, DocumentError):
        st.session_state["flash_error"] = "The demo notes file could not be loaded."


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="sp-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def render_sources(sources) -> None:
    for source in sources:
        st.markdown(f"**{source.label}** - {escape_markdown(source.document)} - similarity {source.score:.2f}")
        st.text(source.excerpt)


def render_document_info(doc: Document) -> None:
    st.subheader("Document information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", doc.name if len(doc.name) <= 22 else doc.name[:19] + "...")
    c2.metric("Pages" if doc.source_type == "pdf" else "Sections", doc.page_count)
    c3.metric("Words", f"{doc.word_count:,}")
    c4.metric("Reading time", f"{estimate_reading_minutes(doc.word_count)} min")
    with st.expander("Preview of extracted text"):
        st.text(doc.text[:1500] + ("..." if len(doc.text) > 1500 else ""))


# ----------------------------------------------------------------------- pages
def page_dashboard(settings: Settings) -> None:
    st.markdown(
        '<div class="sp-hero"><h1>Study smarter with your own notes</h1>'
        "<p>Upload notes, then summarise them, ask questions with page references, practise with quizzes "
        "and review key concepts.</p></div>",
        unsafe_allow_html=True,
    )
    doc: Document | None = st.session_state.doc
    if doc is None:
        st.info("Start by uploading a PDF/TXT file, or load the demo notes to try everything instantly.")
        c1, c2 = st.columns(2)
        c1.button("Upload notes", key="dash_upload", on_click=goto, args=("Upload",), type="primary")
        c2.button("Try demo notes", key="dash_demo", on_click=load_demo)
    else:
        st.success(f"Loaded: {doc.name}")
        render_document_info(doc)

    st.subheader("What you can do")
    cols = st.columns(4)
    items = [
        ("Summary", "Short and detailed summaries built from key sentences in your notes."),
        ("Ask My Notes", "Questions answered from retrieved passages, with page references."),
        ("Quiz", "Multiple-choice practice with scoring and explanations."),
        ("Key Concepts", "Important terms with definitions and where they appear."),
    ]
    for col, (title, body) in zip(cols, items):
        with col:
            card(title, body)
            st.button(f"Open {title}", key=f"open_{title}", on_click=goto, args=(title,))

    st.subheader("How it works")
    st.markdown(
        "1. Text is extracted page by page and cleaned.\n"
        "2. Text is split into overlapping chunks that remember their page.\n"
        "3. Chunks become vectors and are compared with your question using cosine similarity.\n"
        "4. The best passages are used as evidence, and sources are always shown."
    )
    mode = "neural embeddings (Sentence Transformers)" if st.session_state.use_dense else "TF-IDF keyword matching"
    st.caption(
        f"Search mode requested: {mode}. Language model: "
        f"{'on (optional, local)' if st.session_state.use_generative else 'off - extractive answers'}."
    )


def page_upload(settings: Settings) -> None:
    st.header("Upload notes")
    st.write(f"Supported: PDF and TXT, up to {settings.max_file_mb} MB. Files are processed in memory and are not sent anywhere.")
    uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"], key="uploader",
                                help="Scanned PDFs without selectable text are not supported (no OCR).")
    c1, c2 = st.columns(2)
    process = c1.button("Process notes", type="primary", key="process", disabled=uploaded is None)
    c2.button("Use demo notes instead", key="upload_demo", on_click=load_demo)

    if process and uploaded is not None:
        with st.spinner("Reading and cleaning your document..."):
            try:
                doc = load_document(uploaded.name, uploaded.getvalue(), settings)
                set_document(doc)
                st.success(f"Processed {doc.name}: {doc.page_count} {doc.unit_name.lower()}(s), {doc.word_count:,} words.")
            except DocumentError as exc:
                st.error(exc.user_message)
            except Exception:
                logger.exception("Unexpected error while reading an upload")
                st.error("Sorry, this file could not be processed. Please try a different file.")
    if uploaded is not None:
        st.caption(f"Selected: {escape_markdown(uploaded.name)} ({format_file_size(uploaded.size)})")
    if st.session_state.doc is not None:
        render_document_info(st.session_state.doc)
    else:
        st.info("No document loaded yet.")


def page_summary(settings: Settings) -> None:
    st.header("Summary")
    doc = require_document()
    if doc is None:
        return
    generator = current_generator(settings)
    cache_key = (doc.fingerprint, generator is not None)
    if st.button("Generate summary", type="primary", key="gen_summary"):
        with st.spinner("Summarising your notes..." + (" (loading the language model on first use)" if generator else "")):
            try:
                st.session_state.summaries[cache_key] = summarize(doc, generator)
            except Exception:
                logger.exception("Summarisation failed")
                st.error("The summary could not be created for this document.")
    result = st.session_state.summaries.get(cache_key)
    if result is None:
        st.info("Press 'Generate summary' to create a short and a detailed summary.")
        return
    for warning in result.warnings:
        st.warning(warning)
    st.caption(result.method)
    short_tab, detailed_tab = st.tabs(["Short summary", "Detailed summary"])
    with short_tab:
        st.write(escape_markdown(result.short))
    with detailed_tab:
        for point in result.detailed:
            st.markdown(f"- {escape_markdown(point.text)} *({point.label})*")


def page_ask(settings: Settings) -> None:
    st.header("Ask My Notes")
    doc = require_document()
    if doc is None:
        return
    pipeline = ensure_pipeline(settings)
    if pipeline is None:
        return
    if st.session_state.embedder_note:
        st.info(st.session_state.embedder_note)
    st.caption(f"Search engine: {pipeline.embedder.name}. {len(pipeline.chunks)} passages indexed.")

    ask_tab, search_tab = st.tabs(["Ask a question", "Search passages"])
    with ask_tab:
        history = st.container()
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input("Your question", max_chars=300, placeholder="e.g. What is overfitting?")
            asked = st.form_submit_button("Ask")
        if asked and question.strip():
            st.session_state.chat.append({"role": "user", "content": question.strip()})
            with st.spinner("Searching your notes..."):
                try:
                    answer = pipeline.ask(question, current_generator(settings))
                    st.session_state.chat.append({"role": "assistant", "answer": answer})
                except Exception:
                    logger.exception("Question answering failed")
                    st.error("Sorry, I couldn't answer that question. Please try again.")
        elif asked:
            st.warning("Please type a question first.")
        with history:
            if not st.session_state.chat:
                st.info("Ask something about your notes. Answers always show the passages and pages they come from.")
            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.write(escape_markdown(message["content"]))
                        continue
                    answer = message["answer"]
                    (st.write if answer.found else st.warning)(escape_markdown(answer.text))
                    st.caption(answer.method_label)
                    for note in answer.notes:
                        st.caption(note)
                    if answer.sources:
                        with st.expander(f"Source passages ({len(answer.sources)})"):
                            render_sources(answer.sources)
    with search_tab:
        with st.form("search_form"):
            query = st.text_input("Search your notes", max_chars=200, placeholder="e.g. learning rate")
            top_k = st.slider("Number of passages", 1, 8, settings.top_k)
            searched = st.form_submit_button("Search")
        if searched:
            if not query.strip():
                st.warning("Please enter a search phrase.")
            else:
                results = pipeline.search(query, top_k)
                if not results:
                    st.warning("No sufficiently relevant passages were found.")
                for r in results:
                    st.markdown(f"**{r.chunk.label}** - similarity {r.score:.2f}")
                    st.text(r.chunk.text)


def page_quiz(settings: Settings) -> None:
    st.header("Quiz")
    doc = require_document()
    if doc is None:
        return
    count = st.number_input("Number of questions", min_value=3, max_value=10, value=5, step=1, key="quiz_n")
    if st.button("Generate quiz", type="primary", key="gen_quiz"):
        with st.spinner("Creating questions from your notes..."):
            try:
                quiz_set = generate_quiz(doc, int(count), current_generator(settings), seed=int(time.time() * 1000) % 100000)
                st.session_state.quiz_counter += 1
                st.session_state.quiz = {"id": st.session_state.quiz_counter, "set": quiz_set, "score": None}
            except Exception:
                logger.exception("Quiz generation failed")
                st.error("The quiz could not be created for this document.")

    quiz = st.session_state.quiz
    if quiz is None:
        st.info("Press 'Generate quiz' to practise. Correct answers stay hidden until you submit.")
        return
    quiz_set = quiz["set"]
    for warning in quiz_set.warnings:
        st.warning(warning)
    if not quiz_set.questions:
        st.error("These notes do not contain enough definitions or clear facts to build a quiz.")
        return
    st.caption(quiz_set.method)

    if quiz["score"] is None:
        with st.form(f"quiz_form_{quiz['id']}"):
            for i, q in enumerate(quiz_set.questions):
                st.radio(f"Q{i + 1}. {escape_markdown(q.question)}", q.options, index=None,
                         format_func=escape_markdown, key=f"quiz_{quiz['id']}_{i}")
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            answers = {}
            for i, q in enumerate(quiz_set.questions):
                chosen = st.session_state.get(f"quiz_{quiz['id']}_{i}")
                answers[i] = q.options.index(chosen) if chosen in q.options else None
            quiz["score"] = score_quiz(quiz_set.questions, answers)
            st.rerun()
        return

    score = quiz["score"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{score.correct} / {score.total}")
    c2.metric("Percentage", f"{score.percentage:.0f}%")
    c3.metric("Unanswered", score.unanswered)
    (st.success if score.percentage >= 70 else st.info)(score.feedback)
    st.subheader("Review")
    for q, result in zip(quiz_set.questions, score.results):
        with st.container(border=True):
            st.markdown(f"**Q{result.index + 1}.** {escape_markdown(q.question)}")
            mine = q.options[result.selected_index] if result.selected_index is not None else "No answer"
            if result.is_correct:
                st.success(f"Your answer: {mine}")
            else:
                st.error(f"Your answer: {mine}")
                st.write(f"Correct answer: {escape_markdown(q.options[q.correct_index])}")
            st.caption(escape_markdown(q.explanation))
    st.button("Start a new quiz", key="reset_quiz", on_click=lambda: st.session_state.update(quiz=None))


def page_concepts(settings: Settings) -> None:
    st.header("Key Concepts")
    doc = require_document()
    if doc is None:
        return
    limit = st.slider("How many concepts?", 5, 20, 12, key="concept_n")
    with st.spinner("Finding key concepts..."):
        concepts = extract_key_concepts(doc, limit)
    if not concepts:
        st.warning("This document is too short to extract meaningful concepts.")
        return
    if doc.word_count < 120:
        st.caption("Short document: concepts are based on very few mentions, so treat them as rough suggestions.")
    for concept in concepts:
        with st.expander(f"{concept.term}  -  {doc.label(concept.page)}"):
            if concept.definition:
                st.write(f"**Definition from the notes:** {escape_markdown(concept.definition)}")
            else:
                st.caption("No explicit definition found; showing the first supporting sentence.")
            st.text(concept.snippet)


# ------------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title="StudyPilot AI", page_icon="\U0001F9ED", layout="wide")
    settings = get_settings()
    configure_logging(settings.log_level)
    init_state(settings)
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div class="sp-logo"><div class="sp-logo-mark">\U0001F9ED</div><div><div class="sp-logo-text">StudyPilot AI</div>'
            '<div class="sp-logo-sub">Smart notes to learning assistant</div></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Navigate", PAGES, key="nav")
        st.divider()
        st.subheader("Settings")
        st.checkbox("Use neural embeddings (Sentence Transformers)", key="use_dense",
                    help="Better semantic search. Downloads a small model on first use; falls back to TF-IDF if unavailable.")
        st.checkbox("Use optional local language model", key="use_generative",
                    help="Uses Hugging Face Transformers on your CPU. Downloads model weights on first use and is slower.")
        if st.session_state.doc is not None:
            st.caption(f"Loaded: {st.session_state.doc.name}")
            st.button("Clear loaded document", key="clear_doc", on_click=clear_session)
        st.caption(f"StudyPilot AI v{__version__} - your files stay on this machine.")

    if st.session_state.pop("flash", None):
        st.success("Demo notes loaded - explore the other pages from the sidebar.")
    if st.session_state.pop("flash_error", None):
        st.error("The demo notes file could not be loaded.")

    renderers = {"Dashboard": page_dashboard, "Upload": page_upload, "Summary": page_summary,
                 "Ask My Notes": page_ask, "Quiz": page_quiz, "Key Concepts": page_concepts}
    renderers[st.session_state.nav](settings)


main()
