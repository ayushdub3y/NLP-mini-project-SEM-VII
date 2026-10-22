"""Counselling support: crisis screening, coping suggestions, reflective prompts.

This is a supportive aid for a human counsellor, not a diagnostic tool.
"""
import re
import unicodedata
from copy import deepcopy

CRISIS_PATTERNS = [
    r"\bkill (myself|my ?self)\b", r"\bend (my|this) life\b", r"\bsuicid", r"\bwant to die\b",
    r"\bdon'?t want to (live|be alive|exist)\b", r"\bbetter off (dead|without me)\b",
    r"\bno reason to live\b", r"\bhurt(ing)? myself\b", r"\bself[- ]?harm\b", r"\bcut(ting)? myself\b",
    r"\bend it all\b", r"\bcan'?t go on\b",
    r"\b(?:take|end) my (?:own )?life\b",
    r"\b(?:do not|don't|dont) want to (?:live|be alive|exist)\b",
    r"\b(?:wish|hope) (?:that )?i (?:(?:was|were) dead|(?:wasn't|weren't|was not|were not) alive)\b",
    r"\b(?:kill|hurt|cut)(?:ing|ting)? my\s*self\b",
    r"\b(?:want|plan|planning|going) to (?:overdose|harm myself)\b",
    r"\b(?:cannot|can not) go on\b",
]
_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

# Direct users to a country selector rather than embedding unverified numbers.
SUPPORT_DIRECTORY = "https://findahelpline.com/"


def crisis_check(text):
    """Conservative keyword screen, not a risk assessment.

    Negations, quotations and third-person reports can still trigger support.
    Do not suppress them with brittle negation rules; missed risk is costly.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "\u2019": "'", "\u2018": "'", "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-",
    }))
    normalized = re.sub(r"\s+", " ", normalized)
    return bool(_CRISIS_RE.search(normalized))


SUPPORT = {
    "sadness": {
        "validation": "It sounds like you're carrying something heavy. Feeling sad is a natural response to loss or disappointment.",
        "strategies": [
            "Reach out to one person you trust, even with a short message.",
            "Do one small, gentle activity: a short walk, a warm drink, some sunlight.",
            "Allow yourself to feel it. Writing it down can help it feel less overwhelming.",
        ],
        "prompts": ["What do you think is behind this sadness?", "What has helped you feel a little better in the past?"],
    },
    "fear": {
        "validation": "It sounds like you're feeling anxious or afraid. That's your mind trying to protect you.",
        "strategies": [
            "Try box breathing: in for 4, hold 4, out 4, hold 4. Repeat for a few minutes.",
            "Ground yourself: name 5 things you see, 4 you hear, 3 you can touch.",
            "Write down the worry, then separate what you can control from what you can't.",
        ],
        "prompts": ["What exactly are you most afraid might happen?", "How likely is that, and what would you do if it did?"],
    },
    "anger": {
        "validation": "It sounds like you're really frustrated. Anger often points to something that matters to you.",
        "strategies": [
            "Pause before responding. Step away for a few minutes if you can.",
            "Release the energy physically: a brisk walk, stretching, or exercise.",
            "Identify the need underneath the anger (respect, fairness, being heard).",
        ],
        "prompts": ["What happened just before you started feeling this way?", "What do you need right now that you aren't getting?"],
    },
    "disgust": {
        "validation": "It sounds like something has felt deeply wrong or unpleasant to you.",
        "strategies": [
            "Notice what exactly triggered the reaction: a person, an action, a situation.",
            "Create some distance from it for now, physically or mentally.",
            "Consider whether a value of yours was crossed, and how to honour it.",
        ],
        "prompts": ["What about this feels most unacceptable to you?", "What would feeling better look like?"],
    },
    "joy": {
        "validation": "The wording may suggest positive feelings. Does that fit your experience?",
        "strategies": [
            "Notice what contributed to the feeling and what matters to you about it.",
            "Consider how the situation affected you and anyone else involved.",
            "Think about whether what happened fits your values and what you want to do next.",
        ],
        "prompts": ["How do you feel about what happened?", "What effect did it have on you or other people?"],
    },
    "surprise": {
        "validation": "It sounds like something unexpected happened. It can take a while to process.",
        "strategies": [
            "Give yourself time before deciding how you feel or what to do.",
            "Talk it through with someone to make sense of it.",
            "Notice whether the surprise is pleasant, stressful, or mixed.",
        ],
        "prompts": ["How do you feel about it now that it's sunk in?", "What do you need in order to adjust?"],
    },
    "neutral": {
        "validation": "Thanks for sharing. Nothing strong stands out in how this is written.",
        "strategies": [
            "Check in with your body: any tension, tiredness, or hunger?",
            "Try describing the same day with one feeling word.",
        ],
        "prompts": ["Is there anything you're finding hard to put into words?", "How would you like to feel by the end of today?"],
    },
}

# Emotions that suggest a counsellor should look more closely when persistent.
CONCERN_EMOTIONS = {"sadness", "fear", "anger"}


def respond(result):
    """Build a supportive response from a detect() result."""
    if result.get("reading_status") == "needs_context":
        return {
            "validation": "What happened does not tell me enough about how you feel. You may feel one way, several ways, or have no strong feeling about it.",
            "strategies": ["Add a sentence about how you feel now, in your own words.",
                           "Separate what happened from your reaction to it.",
                           "Consider what the experience meant to you and anyone else involved."],
            "prompts": ["How did that feel for you?", "What would you like to reflect on about it?"],
        }
    if result.get("uncertain", False):
        return {
            "validation": "Your entry may contain mixed feelings. A single label may not capture what you mean.",
            "strategies": ["Name the feelings that fit your experience in your own words.",
                           "Consider talking through the entry with someone you trust."],
            "prompts": ["Which feeling words fit best for you?", "What would support you right now?"],
        }
    info = SUPPORT[result["top"]]
    return deepcopy(info)


def trend_flag(history, window=5, threshold=0.6):
    """Flag if most recent entries are dominated by distressing emotions."""
    if not isinstance(window, int) or isinstance(window, bool) or window < 3:
        raise ValueError("window must be an integer of at least 3.")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between 0 (exclusive) and 1.")
    recent = [entry for entry in history[-window:]
              if entry.get("top") in SUPPORT and not entry.get("uncertain", False)]
    if len(recent) < 3:
        return None
    share = sum(1 for e in recent if e["top"] in CONCERN_EMOTIONS) / len(recent)
    if share >= threshold:
        return (f"{round(share * 100)}% of {len(recent)} classified entries in the recent window suggest sadness, fear or anger. "
                "Consider a follow-up conversation.")
    return None
