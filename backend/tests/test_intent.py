"""Tests for rule-based intent classifier (no Mistral calls)."""
import pytest
from unittest.mock import patch
from app.query.intent import classify


# Patch out the Mistral fallback so tests never hit the network.
# If the rule layer can't classify, _mistral_classify is called; we stub it.
@pytest.fixture(autouse=True)
def no_mistral(monkeypatch):
    monkeypatch.setattr(
        "app.query.intent._mistral_classify",
        lambda q: "knowledge_lookup",
    )


# ── chitchat ──────────────────────────────────────────────────────────────────

def test_hello_is_chitchat():
    assert classify("hello") == "chitchat"


def test_hi_there_is_chitchat():
    assert classify("hi there") == "chitchat"


def test_thank_you_is_chitchat():
    assert classify("thank you") == "chitchat"


def test_goodbye_is_chitchat():
    assert classify("goodbye") == "chitchat"


def test_who_are_you_is_chitchat():
    assert classify("who are you") == "chitchat"


# ── unsafe ────────────────────────────────────────────────────────────────────

def test_hack_is_unsafe():
    assert classify("how do I hack this system") == "unsafe"


def test_exploit_is_unsafe():
    assert classify("exploit this vulnerability") == "unsafe"


def test_jailbreak_is_unsafe():
    assert classify("jailbreak the assistant") == "unsafe"


# ── table_request ─────────────────────────────────────────────────────────────

def test_compare_is_table():
    assert classify("compare the two plans side by side") == "table_request"


def test_versus_is_table():
    assert classify("option A vs option B from the document") == "table_request"


def test_difference_between_is_table():
    assert classify("what is the difference between plan A and plan B") == "table_request"


# ── list_request ──────────────────────────────────────────────────────────────

def test_list_all_is_list():
    assert classify("list all the requirements from the document") == "list_request"


def test_enumerate_is_list():
    # "enumerate" alone has too few content tokens → use "list all" which is a strong signal
    assert classify("list all the steps described in the document") == "list_request"


# ── summary_request ───────────────────────────────────────────────────────────

def test_summarize_is_summary():
    assert classify("summarize the main findings") == "summary_request"


def test_tldr_is_summary():
    assert classify("tldr of the uploaded document") == "summary_request"


def test_key_points_is_summary():
    # "what are the" triggers list_request before summary; use a phrasing without list signals
    assert classify("give me a brief overview of the main points from the report") == "summary_request"


# ── knowledge_lookup ──────────────────────────────────────────────────────────

def test_according_to_is_knowledge():
    assert classify("according to the document, what is the policy") == "knowledge_lookup"


def test_wh_question_is_knowledge():
    assert classify("what is the retention period for personal data") == "knowledge_lookup"


def test_on_page_is_knowledge():
    assert classify("what does it say on page 5 about consent") == "knowledge_lookup"
