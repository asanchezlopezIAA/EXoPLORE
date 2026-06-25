"""Basic import and version test."""

def test_import_exoplore():
    import exoplore
    # Version is set in src/exoplore/__init__.py
    assert exoplore.__version__ == "0.2.0"

def test_exoplore_has_version():
    import exoplore
    assert hasattr(exoplore, "__version__")
