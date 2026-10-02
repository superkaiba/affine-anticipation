"""Display names for the Section 4.2 per-element answer-shift rows.

One source of truth for the nine main rows and the three appendix slot rows the
Section 4.2 figures print.  In this release ``figures/section42.py`` is the only
consumer: it imports the row names to build its group tuples and calls
:func:`tick_label` to format each y-tick.

The row string does double duty.  The rebuild step that produced
``records/issue_2564/section42_element_shifts.json`` wrote it into that file as
the ``row`` field, and the renderer keys its group tuples off that same string
before printing it as a y-tick label.  Join key and display label are the same
object, so renaming one without the other silently drops a row from a figure.
Importing these constants moves both from one place: a rename lands here, the
rebuild step rewrites the banked JSON, and the renderer picks up the new label
and the new key together.

Why a sibling module rather than the plotting-style module: that module imports
matplotlib, PIL and ``font_manager`` at import time, and the rebuild step is a
pure torch/numpy pass that draws nothing.  Keeping the names in a module with no
third-party imports lets the generator hold the same constants as the renderer
without pulling the plotting stack into a data rebuild.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Main panel rows
# ---------------------------------------------------------------------------

TONE = "Tone"
PERSONA = "Persona"
OUTPUT_FORMAT = "Output format"
QUESTION_TOPIC = "Question topic"
ONE_WORD_TOPIC = "One-word topic"

# The four refusal rows name what the paired contexts change after the colon,
# and what the model then did before it.
#
# A CONTENT EDIT substitutes one word, so the request itself turns benign or
# harmful: the #2617 bank.  A FRAMING EDIT holds the request fixed and changes
# only how it is put -- past tense, passive voice, declarative curiosity: the
# #2356 bank.
#
# REFUSE/COMPLY means one member of the pair was refused and the other was
# answered.  SAME DECISION means both members got the same decision, refused or
# answered alike.
REFUSAL_REVERSES_INTENT = "Refuse/comply: content edit"
REFUSAL_REVERSES_FRAMING = "Refuse/comply: framing edit"
REFUSAL_HOLDS_INTENT = "Same decision: content edit"
REFUSAL_HOLDS_FRAMING = "Same decision: framing edit"

# ---------------------------------------------------------------------------
# Appendix slot rows: the one-word topic swap split by the grammatical slot the
# changed word occupies.
# ---------------------------------------------------------------------------

SLOT_OBJECT = "slot: object"
SLOT_SUBJECT = "slot: subject"
SLOT_VERB = "slot: verb"


def tick_label(row: str, n_pairs: int | None = None) -> str:
    """A row label formatted for a y-tick column, with its pair count.

    A refusal row names the edit after a colon, which makes it about twice the
    width of the identity rows.  Set on one line it runs off the left edge of a
    full-width canvas, and widening the label gutter to hold it would take the
    width from the panels.  Breaking after the leading clause puts the
    decision contrast (refuse/comply against same decision) on the first line
    and the edit on the second, so the block of refusal rows stays inside a
    gutter narrower than the one the single-line identity rows already need.
    Rows with no colon are unchanged.
    """

    suffix = "" if n_pairs is None else f" (n={n_pairs})"
    head, sep, tail = row.partition(":")
    if not sep:
        return f"{row}{suffix}"
    return f"{head}:\n{tail.strip()}{suffix}"


WRAPPED_TICK_LINESPACING = 0.92
"""Leading inside a wrapped row label.

Tighter than the matplotlib default so the two lines of one label bind into a
single block and separate from the label above and below.
"""

WRAPPED_ROW_PITCH_IN = 0.48
"""Plot height per row, in inches, that a wrapped label needs to clear its neighbours.

A two-line label at the pinned tick size measures about 0.44 in tall, so the
one-line pitch these figures used before the refusal rows took a two-line label
(0.32 in to 0.40 in depending on the figure) overlapped adjacent labels by
0.08 in to 0.17 in.  Figures that draw one row per element size their canvas from this
constant instead of hard-coding a pitch, so all three carry the same row rhythm.
"""
