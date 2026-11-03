"""Auxiliary reading of chat abbreviations. Original question remains authoritative."""
import re
import unicodedata

# Ambiguous k/dt/th/kh may denote amounts or entity codes: leave them alone.
ALIASES = {"hqua": "hôm qua", "hnay": "hôm nay", "bnhiu": "bao nhiêu",
           "nhiu": "nhiều", "ko": "không", "khum": "không", "trc": "trước", "dthu": "doanh thu"}


def auxiliary_question(question):
    text = unicodedata.normalize("NFC", question)
    return re.sub(r"(?<![\w])(?:" + "|".join(ALIASES) + r")(?![\w])", lambda m: ALIASES[m[0]], text)


def question_hint(question):
    other = auxiliary_question(question)
    return "" if other == question else ("\nAuxiliary reading (may be ambiguous; original question controls):\n" + other)
