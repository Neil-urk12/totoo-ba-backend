def normalize_string(text: str) -> str:
    """Normalize string for comparison: uppercase and remove spaces."""
    if not text:
        return ""
    return text.upper().replace(" ", "")