from bs4 import BeautifulSoup
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import re

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"



def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML, URLs, emails, and extra whitespace.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Lowercase & strip
    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    return text
