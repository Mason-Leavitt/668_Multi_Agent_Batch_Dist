DETAIL_KEYWORDS = (
    "i don't understand",
    "dont understand",
    "do not understand",
    "explain",
    "i'm lost",
    "im lost",
    "what does that mean",
    "walk me through it",
    "walk me through",
    "huh",
    "why",
    "what?",
)


def wants_detailed_explanation(user_message: str) -> bool:
    text = user_message.strip().lower()
    return any(keyword in text for keyword in DETAIL_KEYWORDS)
