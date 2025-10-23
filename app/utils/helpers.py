"""Utility helper functions for string manipulation and data processing."""


def normalize_string(text: str) -> str:
    """Normalize string for comparison by converting to uppercase and removing spaces.

    Used for case-insensitive and whitespace-insensitive string comparisons,
    particularly useful for matching product IDs and registration numbers.

    Args:
        text: The string to normalize.

    Returns:
        str: Normalized string in uppercase with spaces removed.

    Example:
        >>> normalize_string("BR-1234 ABC")
        'BR-1234ABC'
    """
    if not text:
        return ""
    return text.upper().replace(" ", "")
