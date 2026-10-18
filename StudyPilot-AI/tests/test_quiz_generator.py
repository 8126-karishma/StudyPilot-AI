import random

import pytest

from studypilot.quiz_generator import (
    QuizQuestion, deduplicate, generate_fallback_questions, generate_quiz, is_grounded,
    parse_model_question, score_quiz, validate_question,
)

from conftest import BrokenGenerator, FakeGenerator


def good(**overrides) -> QuizQuestion:
    base = dict(question="Which gas do plants take in?", options=["Oxygen", "Carbon dioxide", "Helium", "Neon"],
                correct_index=1, explanation="Plants absorb carbon dioxide.", source_page=1, source_text="...")
    base.update(overrides)
    return QuizQuestion(**base)


def test_valid_question_passes():
    assert validate_question(good()) == []


@pytest.mark.parametrize("overrides", [
    {"question": ""}, {"question": "short"}, {"options": ["only one"]},
    {"options": ["a", "b", "c", "d", "e"]}, {"options": ["Same", "same", "x", "y"]},
    {"options": ["a", "", "c", "d"]}, {"correct_index": 4}, {"correct_index": -1},
    {"explanation": "  "},
])
def test_invalid_questions_are_flagged(overrides):
    assert validate_question(good(**overrides))


def test_deduplicate_removes_repeats():
    q1, q2 = good(), good(question="WHICH gas do plants take in??")
    q3 = good(question="Another question about plants?", options=["Oxygen", "Water", "Helium", "Neon"],
              correct_index=1)
    assert deduplicate([q1, q2, q3]) == [q1, q3]


def test_fallback_questions_are_valid_unique_and_supported_by_source(sample_doc):
    questions = generate_fallback_questions(sample_doc, 10, random.Random(3))
    assert len(questions) >= 8
    assert all(validate_question(q) == [] for q in questions)
    assert len({q.question for q in questions}) == len(questions)
    for q in questions:
        correct = q.options[q.correct_index].rstrip(".")
        assert correct.lower() in q.source_text.lower(), (q.question, correct)
        assert q.source_text in sample_doc.text
        assert q.source_page in {p.number for p in sample_doc.pages}
        assert len(q.options) >= 3


def test_fallback_is_deterministic_for_a_seed(sample_doc):
    a = generate_quiz(sample_doc, 5, seed=11)
    b = generate_quiz(sample_doc, 5, seed=11)
    assert [q.question for q in a.questions] == [q.question for q in b.questions]
    assert len(a.questions) == 5 and "no language model" in a.method


def test_quiz_for_tiny_document_reports_shortage():
    from studypilot.text_processor import Document, PageText

    doc = Document("tiny.txt", "txt", [PageText(1, "Hello there.")])
    quiz = generate_quiz(doc, 5, seed=1)
    assert quiz.questions == [] and quiz.warnings


def test_parse_model_question_variants():
    raw = '```json\n{"question": "What is gravity exactly?", "options": ["A force", "A color", "A food", "A song"], "answer": "A", "explanation": "x"}\n```'
    q = parse_model_question(raw, "Gravity is a force.", 2)
    assert q and q.correct_index == 0 and q.source_page == 2
    by_text = parse_model_question('{"question": "What is gravity exactly?", "options": ["A force", "A color"], "answer": "a color"}')
    assert by_text and by_text.correct_index == 1
    by_int = parse_model_question('{"question": "What is gravity exactly?", "options": ["A force", "A color"], "answer": 0}')
    assert by_int and by_int.correct_index == 0


@pytest.mark.parametrize("raw", [
    "", "no json here", "{not valid json}", '{"question": "Q?"}', '{"question": "What is it exactly?", "options": "abc", "answer": "A"}',
    '{"question": "What is it exactly?", "options": ["x", "y"], "answer": "Z"}',
    '{"question": "What is it exactly?", "options": ["x", "x"], "answer": "A"}',
    '{"question": "What is it exactly?", "options": ["x", "y"], "answer": true}',
    '{"question": "What is it exactly?", "options": ["x", "y"], "answer": 9}', "[1, 2, 3]",
])
def test_malformed_model_output_is_rejected(raw):
    assert parse_model_question(raw) is None


def test_grounding_check():
    assert is_grounded("a force between masses", "Gravity is a force between masses.")
    assert not is_grounded("a delicious pasta dish", "Gravity is a force between masses.")
    assert not is_grounded("the and of", "anything")


def test_model_questions_used_only_when_valid_and_grounded():
    from studypilot.text_processor import Document, PageText

    passage = ("Chlorophyll is the green pigment in plant cells that absorbs light energy from the sun. "
               "This energy is then used to turn water and carbon dioxide into glucose and oxygen during "
               "the process called photosynthesis, which happens mostly inside the leaves of the plant.")
    doc = Document("bio.txt", "txt", [PageText(1, passage)])
    reply = ('{"question": "What does chlorophyll do for the plant?", "options": '
             '["absorbs light energy", "stores water", "makes seeds", "builds roots"], '
             '"answer": "A", "explanation": "Stated in the notes."}')
    quiz = generate_quiz(doc, 1, FakeGenerator(reply), seed=1)
    assert quiz.questions[0].question == "What does chlorophyll do for the plant?"
    assert "Local model" in quiz.method and quiz.questions[0].source_page == 1

    ungrounded = ('{"question": "What is the capital city of France?", "options": ["Paris city", "Rome city"], '
                  '"answer": "A", "explanation": "x"}')
    quiz = generate_quiz(doc, 1, FakeGenerator(ungrounded), seed=1)
    assert all("France" not in q.question for q in quiz.questions)
    assert any("discarded" in w for w in quiz.warnings)


def test_broken_generator_falls_back(sample_doc):
    quiz = generate_quiz(sample_doc, 4, BrokenGenerator(), seed=2)
    assert len(quiz.questions) == 4 and any("rule-based" in w for w in quiz.warnings)


def test_malformed_generator_output_still_produces_quiz(sample_doc):
    quiz = generate_quiz(sample_doc, 4, FakeGenerator("total garbage"), seed=2)
    assert len(quiz.questions) == 4


def test_scoring_all_correct_partial_and_missing():
    questions = [good(question=f"Question number {i} about plants?", options=["a", "b", "c", "d"], correct_index=i % 4,
                      source_text=str(i)) for i in range(4)]
    perfect = score_quiz(questions, {i: i % 4 for i in range(4)})
    assert (perfect.correct, perfect.total, perfect.percentage) == (4, 4, 100.0)
    assert "Excellent" in perfect.feedback

    partial = score_quiz(questions, {0: 0, 1: 0})  # 1 right, 1 wrong, 2 missing
    assert (partial.correct, partial.answered, partial.unanswered) == (1, 2, 2)
    assert partial.percentage == 25.0 and partial.results[3].selected_index is None

    none = score_quiz(questions, {})
    assert none.correct == 0 and none.unanswered == 4 and none.percentage == 0.0


def test_scoring_ignores_invalid_selections_and_empty_quiz():
    questions = [good()]
    result = score_quiz(questions, {0: 99})
    assert result.correct == 0 and result.unanswered == 1
    assert score_quiz(questions, {0: None}).answered == 0
    assert score_quiz(questions, {0: True}).correct == 0
    empty = score_quiz([], {})
    assert empty.total == 0 and empty.percentage == 0.0 and "no questions" in empty.feedback.lower()


def test_score_percentage_rounding_and_feedback_tiers():
    qs = [good(question=f"Question number {i} about plants?", source_text=str(i)) for i in range(3)]
    result = score_quiz(qs, {0: 1, 1: 1, 2: 0})
    assert result.percentage == 66.7 and "Fair" in result.feedback
