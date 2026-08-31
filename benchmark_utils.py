import re

_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_reference_answer(text: str) -> str:
    answer = _last_number(text.rpartition("####")[2])
    if answer is None:
        raise ValueError("GSM8K reference answer is missing its final number")
    return answer


def extract_predicted_answer(text: str) -> str | None:
    return _last_number(text)
