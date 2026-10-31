import re


def slugify(text: str, max_len: int = 50) -> str:
    text = text.lower()
    # Keep CJK characters along with a-z and 0-9
    text = re.sub(r'[^\u4e00-\u9fff\uac00-\ud7af\u3130-\u318f\ua960-\ua97f\w]+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    text = text.strip('-')
    return text[:max_len]

