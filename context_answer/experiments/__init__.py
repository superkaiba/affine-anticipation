"""Per-experiment entry-point packages.

Each subpackage corresponds to a single issue/task (one experiment per
subpackage). Shared library code lives in the surrounding
`context_answer` package; experiment subpackages import from it for
training, eval, persona, and data helpers.
"""
