"""Headless UI smoke tests using Streamlit's AppTest (no browser, no model downloads)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("STUDYPILOT_USE_DENSE_EMBEDDINGS", "false")
    monkeypatch.setenv("STUDYPILOT_USE_GENERATIVE_MODELS", "false")
    return AppTest.from_file(APP, default_timeout=90).run()


def goto(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_renders_with_empty_state(app):
    assert not app.exception
    assert any("Start by uploading" in i.value for i in app.info)


def test_every_page_shows_empty_state_without_document(app):
    for page in ["Summary", "Ask My Notes", "Quiz", "Key Concepts"]:
        goto(app, page)
        assert any("No notes loaded" in i.value for i in app.info), page


def test_demo_flow_all_pages(app):
    app.button(key="dash_demo").click().run()
    assert not app.exception and any("Loaded: sample_notes.txt" in s.value for s in app.success)

    goto(app, "Summary")
    app.button(key="gen_summary").click().run()
    assert not app.exception and len(app.tabs) == 2

    goto(app, "Ask My Notes")
    app.text_input[0].set_value("What is overfitting?")
    form_button = [b for b in app.button if b.label == "Ask"][0]
    form_button.click().run()
    assert not app.exception
    assert any("TF-IDF" in i.value for i in app.info)

    goto(app, "Key Concepts")
    assert not app.exception and len(app.expander) >= 5


def test_quiz_hides_answers_until_submit_then_scores(app):
    app.button(key="dash_demo").click().run()
    goto(app, "Quiz")
    app.button(key="gen_quiz").click().run()
    quiz_radios = [r for r in app.radio if r.key and r.key.startswith("quiz_")]
    assert len(quiz_radios) == 5
    page_text = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
    assert "The notes say" not in page_text and "Correct answer" not in page_text
    assert not app.metric  # no score before submission

    quiz_radios[0].set_value(quiz_radios[0].options[0])
    [b for b in app.button if b.label == "Submit answers"][0].click().run()
    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Score"].endswith("/ 5") and labels["Unanswered"] == "4"
