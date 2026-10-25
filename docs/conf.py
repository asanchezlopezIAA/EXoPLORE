import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
autodoc_mock_imports = ["numpy", "scipy", "astropy", "matplotlib"]
autodoc_use_legacy_class_based = True

project = "EXoPLORE"
author = "Alejandro Sánchez-López"
copyright = "2026, Alejandro Sánchez-López"
release = "1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]
exclude_patterns.append("czesla2024_pipeline.md")


def _validate_notebook_download(path):
    """Validate the saved notebook outputs before Sphinx copies the download."""
    import base64
    import json

    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    errors = [output for cell in code for output in cell.get("outputs", [])
              if output.get("output_type") == "error"]
    counts = [cell.get("execution_count") for cell in code]
    data = [output.get("data", {}) for cell in code
            for output in cell.get("outputs", [])]
    pngs = [item["image/png"] for item in data if "image/png" in item]
    gifs = [item["image/gif"] for item in data if "image/gif" in item]
    reasons = []
    if errors:
        reasons.append(f"{len(errors)} stored error output(s)")
    if counts != list(range(1, len(code) + 1)):
        reasons.append("execution counters are not consecutive in cell order")
    if len(pngs) < 5 or len(gifs) != 1:
        reasons.append("expected at least five embedded figures and one embedded GIF")
    for mime, values, signature in [("PNG", pngs, b"\x89PNG\r\n\x1a\n"),
                                    ("GIF", gifs, b"GIF8")]:
        for encoded in values:
            try:
                raw = base64.b64decode("".join(encoded), validate=True)
                if not raw.startswith(signature):
                    reasons.append(f"invalid embedded {mime}")
            except (ValueError, TypeError):
                reasons.append(f"invalid embedded {mime}")
    for marker, mime in [("animate_ccf_search(", "image/gif"),
                         ("fig = tutorial.plot_detection_map(", "image/png")]:
        matching = [cell for cell in code if marker in "".join(cell["source"])]
        if not matching or not any(mime in output.get("data", {})
                                   for cell in matching for output in cell.get("outputs", [])):
            reasons.append(f"missing saved output for {marker}")
    if "/Users/" in path.read_text(encoding="utf-8"):
        reasons.append("saved artifact contains an absolute /Users/ path")
    if reasons:
        from sphinx.errors import SphinxError
        raise SphinxError("Notebook download failed publication checks: " + "; ".join(reasons))


def _check_notebook_download(app):
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "notebooks/01_first_ANDES_simulation.ipynb"
    _validate_notebook_download(path)


def setup(app):
    # The ordinary MyST download role copies the notebook after this check.
    app.connect("builder-inited", _check_notebook_download)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
