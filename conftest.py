"""Put the repository root on ``sys.path`` for the test suite.

``figures`` is a collection of scripts run from the repository root rather than
an installed package (see ``[tool.setuptools.packages.find]`` in
``pyproject.toml``), so ``tests/test_figures.py`` needs the root importable.
pytest inserts the rootdir of the first conftest it finds, which is what this
empty file is for.
"""
