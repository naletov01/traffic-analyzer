"""Plate text normalization and optional Ukrainian-format validation.

Ukrainian plates are two letters, four digits, two letters (e.g. "AA1234BB"),
using only the Cyrillic letters with Latin look-alikes, which OCR reads as Latin.
"""

import re

# Latin look-alikes of the letters allowed on UA plates.
_UA_LETTERS = "ABEIKMHOPCTX"
_UA_RE = re.compile(rf"^[{_UA_LETTERS}]{{2}}\d{{4}}[{_UA_LETTERS}]{{2}}$")

# Common OCR confusions, applied position-aware (digit slot vs letter slot).
_TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}


def normalize(text: str) -> str:
    """Uppercase and keep only A-Z/0-9 (drops spaces, dashes, EU band, etc.)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _coerce_ua(text: str) -> str:
    """Best-effort fix to the AA1234BB shape using position-aware substitutions."""
    if len(text) != 8:
        return text
    out = []
    for i, ch in enumerate(text):
        if 2 <= i <= 5:                       # digit slots
            out.append(_TO_DIGIT.get(ch, ch))
        else:                                 # letter slots
            out.append(_TO_LETTER.get(ch, ch))
    return "".join(out)


def is_valid_ua(text: str) -> bool:
    return bool(_UA_RE.match(text))


def format_plate(raw: str, mode: str = "none") -> tuple[str | None, bool]:
    """Return (display_text, is_valid).

    mode="none": just normalize; always 'valid' if non-empty.
    mode="ua":   normalize, coerce to UA shape, validate against the UA regex.
    """
    text = normalize(raw)
    if not text:
        return None, False
    if mode != "ua":
        return text, True
    text = _coerce_ua(text)
    return text, is_valid_ua(text)
