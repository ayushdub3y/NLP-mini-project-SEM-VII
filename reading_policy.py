"""Conservative applicability checks for personal emotion feedback.

This is a transparent linguistic heuristic, not another trained model. It may
abstain on implicit feelings. It never changes the transformer's probabilities.
"""
import re
import unicodedata

_FEELINGS = (
    r"angry|anger|furious|frustrated|frustrating|irritated|annoyed|livid|"
    r"disgusted|disgusting|revolting|repulsed|"
    r"afraid|scared|terrified|fearful|anxious|nervous|worried|worrying|dread|"
    r"happy|happier|happiness|joy|joyful|delighted|excited|proud|overjoyed|relieved|grateful|"
    r"sad|sadder|sadness|unhappy|lonely|alone|heartbroken|empty|disappointed|grieving|"
    r"surprised|astonished|stunned|shocked|confused|overwhelmed|stressed|"
    r"guilty|ashamed|regretful|embarrassed|numb|calm|content|disconnected|upset"
)
_AFFECT = re.compile(rf"\b(?:{_FEELINGS})\b")
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|my|me|myself)\b")
_OTHER_PERSON = re.compile(
    r"\b(?:he|she|they|(?:my|a|the) (?:friend|brother|sister|mother|father|partner|teacher))\s+"
    r"(?:is|was|were|feels?|felt|seems?|looks?|looked|has|had|became|got|said)\b"
)
_NEGATED = re.compile(rf"\b(?:not|never|no longer|don't|do not|isn't|wasn't)\s+(?:\w+\s+){{0,2}}(?:{_FEELINGS})\b")
_IMPLICIT_PERSONAL = re.compile(
    r"\bi (?:really |deeply |so )?(?:miss|hate|love|regret|resent|fear|worry)\b|"
    r"\bi (?:am |was |can't stop |couldn't stop |keep )?(?:crying|smiling|sobbing)\b"
)


def assess(text):
    """Return applicability/reason without pretending event valence is emotion."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()
    clauses = re.split(r"[.!?;\n]+|\b(?:but|and|although|whereas)\b", normalized)
    evidence = []
    negated = False
    for clause in clauses:
        # Negation handling intentionally errs toward asking rather than flipping
        # sadness to joy: 'not sad' does not establish happiness.
        if _NEGATED.search(clause):
            negated = True
            continue
        if _OTHER_PERSON.search(clause):
            continue
        personal = _FIRST_PERSON.search(clause) or re.match(r"^\s*(?:feel|felt|feeling)\b", clause)
        if (personal and _AFFECT.search(clause)) or _IMPLICIT_PERSONAL.search(clause):
            evidence.append(clause.strip())
    if negated:
        return {"status": "needs_context", "reason": "negated_feeling", "evidence": evidence}
    if not evidence:
        return {"status": "needs_context", "reason": "no_personal_feeling", "evidence": []}
    # Another person's feeling can dominate the document classifier. Do not
    # attribute that document-level score to the author without clarification.
    if _OTHER_PERSON.search(normalized) and _AFFECT.search(normalized):
        return {"status": "needs_context", "reason": "other_person_context", "evidence": evidence}
    return {"status": "supported", "reason": "personal_feeling", "evidence": evidence}
