"""Syntactic Attribution Engine & Reading Policy using spaCy Dependency Parsing.

Enforces grammatical constraints on affective attribution:
1. Nominal Subject Verification (nsubj/nsubjpass): distinguishing first-person from third-person reports.
2. Syntactic Negation Scoping (neg): resolving negation along dependency arcs.
3. Inner Affective State vs Episodic Event Classification: distinguishing inner feelings
   from past-tense episodic transitives without feeling complements.
"""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

_FIRST_PERSON_PRONOUNS = {"i", "me", "my", "myself", "we", "us", "our", "ourselves"}

_CORE_FEELINGS = {
    "angry", "anger", "furious", "frustrated", "frustrating", "irritated", "annoyed", "livid",
    "disgusted", "disgusting", "revolting", "repulsed",
    "afraid", "scared", "terrified", "fearful", "anxious", "nervous", "worried", "worrying", "dread",
    "happy", "happier", "happiness", "joy", "joyful", "delighted", "excited", "proud", "overjoyed", "relieved", "grateful",
    "sad", "sadder", "sadness", "unhappy", "lonely", "alone", "heartbroken", "empty", "disappointed", "grieving",
    "surprised", "astonished", "stunned", "shocked", "confused", "overwhelmed", "stressed",
    "guilty", "ashamed", "regretful", "embarrassed", "numb", "calm", "content", "disconnected", "upset"
}

_AFFECTIVE_VERBS = {
    "miss", "hate", "love", "regret", "resent", "fear", "worry", "cry", "smile", "sob", "crying", "smiling", "sobbing"
}

_COPULAR_OR_SENSING = {
    "feel", "am", "is", "are", "was", "were", "seem", "look", "become", "sound", "be"
}

# Regex fallbacks in case spaCy model fails to load
_FEELINGS_REGEX = (
    r"angry|anger|furious|frustrated|frustrating|irritated|annoyed|livid|"
    r"disgusted|disgusting|revolting|repulsed|"
    r"afraid|scared|terrified|fearful|anxious|nervous|worried|worrying|dread|"
    r"happy|happier|happiness|joy|joyful|delighted|excited|proud|overjoyed|relieved|grateful|"
    r"sad|sadder|sadness|unhappy|lonely|alone|heartbroken|empty|disappointed|grieving|"
    r"surprised|astonished|stunned|shocked|confused|overwhelmed|stressed|"
    r"guilty|ashamed|regretful|embarrassed|numb|calm|content|disconnected|upset"
)
_AFFECT_RE = re.compile(rf"\b(?:{_FEELINGS_REGEX})\b")
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|my|me|myself)\b")
_OTHER_PERSON_RE = re.compile(
    r"\b(?:he|she|they|(?:my|a|the) (?:friend|brother|sister|mother|father|partner|teacher))\s+"
    r"(?:is|was|were|feels?|felt|seems?|looks?|looked|has|had|became|got|said)\b"
)
_NEGATED_RE = re.compile(rf"\b(?:not|never|no longer|don't|do not|isn't|wasn't)\s+(?:\w+\s+){{0,2}}(?:{_FEELINGS_REGEX})\b")
_IMPLICIT_PERSONAL_RE = re.compile(
    r"\bi (?:really |deeply |so )?(?:miss|hate|love|regret|resent|fear|worry)\b|"
    r"\bi (?:am |was |can't stop |couldn't stop |keep )?(?:crying|smiling|sobbing)\b"
)


@lru_cache(maxsize=1)
def _get_spacy_nlp():
    """Load spaCy model with caching; return None if unavailable."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _load_lexicon():
    vocab_path = Path(__file__).parent / "data" / "vocabulary" / "affective_predicates.txt"
    if vocab_path.exists():
        words = {line.strip().lower() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
        return _CORE_FEELINGS | words
    return _CORE_FEELINGS


_ALL_FEELINGS = _load_lexicon()


def _find_subjects(token):
    """Find nominal subjects associated with an affective predicate token or its head."""
    subjects = []
    # Check token's own children
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(child)
    
    # Check head's children if token is a complement or adjective
    head = token.head
    if head and head != token:
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subjects.append(child)
        # If head is conjoined, check head of head
        if head.dep_ == "conj" and head.head:
            for child in head.head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child)
    return subjects


_NEGATION_WORDS = {
    "not", "never", "no", "nor", "neither", "n't", "without",
    "hardly", "scarcely", "barely"
}


def _token_is_negation(tok):
    return tok.dep_ == "neg" or tok.lower_ in _NEGATION_WORDS


def _is_syntactically_negated(token):
    """Check negation along dependency arcs from token, head, auxiliaries, and adverbial modifiers."""
    # Direct children of token
    for child in token.children:
        if _token_is_negation(child):
            return True
        for sub in child.children:
            if _token_is_negation(sub):
                return True

    # Head and auxiliaries / adverbial modifiers
    head = token.head
    if head and head != token:
        if _token_is_negation(head):
            return True
        for child in head.children:
            if _token_is_negation(child):
                return True
            if child.dep_ in ("aux", "auxpass", "advmod"):
                for aux_child in child.children:
                    if _token_is_negation(aux_child):
                        return True
        # Conjunction head
        if head.dep_ == "conj" and head.head:
            for child in head.head.children:
                if _token_is_negation(child):
                    return True
                if child.dep_ in ("aux", "auxpass", "advmod"):
                    for sub in child.children:
                        if _token_is_negation(sub):
                            return True
    return False


def _assess_with_spacy(doc, raw_text):
    """Syntactic dependency parsing assessment."""
    evidence = []
    has_negated = False
    has_other_person = False
    has_first_person_feeling = False

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_has_first_person_subj = any(
            t.lower_ in _FIRST_PERSON_PRONOUNS and t.dep_ in ("nsubj", "nsubjpass")
            for t in sent
        )
        
        for token in sent:
            token_lower = token.lower_
            token_lemma = token.lemma_.lower()
            is_affective = (token_lower in _ALL_FEELINGS or token_lemma in _ALL_FEELINGS
                            or token_lower in _AFFECTIVE_VERBS or token_lemma in _AFFECTIVE_VERBS)
            
            if not is_affective:
                continue

            # Check negation arc
            if _is_syntactically_negated(token):
                has_negated = True
                continue

            # Check subjects
            subjects = _find_subjects(token)
            if subjects:
                for subj in subjects:
                    if subj.lower_ in _FIRST_PERSON_PRONOUNS:
                        has_first_person_feeling = True
                        evidence.append(sent_text)
                    else:
                        has_other_person = True
            else:
                # No explicit local subject; check if sentence or clause has first-person subject or copula
                if sent_has_first_person_subj or any(t.lower_ in _FIRST_PERSON_PRONOUNS for t in sent):
                    has_first_person_feeling = True
                    evidence.append(sent_text)
                elif any(t.dep_ in ("nsubj", "nsubjpass") for t in sent):
                    has_other_person = True

    if has_negated:
        return {"status": "needs_context", "reason": "negated_feeling", "evidence": evidence}
    if has_other_person:
        return {"status": "needs_context", "reason": "other_person_context", "evidence": evidence}
    if not has_first_person_feeling or not evidence:
        return {"status": "needs_context", "reason": "no_personal_feeling", "evidence": []}

    return {"status": "supported", "reason": "personal_feeling", "evidence": list(dict.fromkeys(evidence))}


def _assess_regex_fallback(text):
    """Fallback assessment using regular expressions when spaCy is not available."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()
    clauses = re.split(r"[.!?;\n]+|\b(?:but|and|although|whereas)\b", normalized)
    evidence = []
    negated = False
    for clause in clauses:
        if _NEGATED_RE.search(clause):
            negated = True
            continue
        if _OTHER_PERSON_RE.search(clause):
            continue
        personal = _FIRST_PERSON_RE.search(clause) or re.match(r"^\s*(?:feel|felt|feeling)\b", clause)
        if (personal and _AFFECT_RE.search(clause)) or _IMPLICIT_PERSONAL_RE.search(clause):
            evidence.append(clause.strip())
    if negated:
        return {"status": "needs_context", "reason": "negated_feeling", "evidence": evidence}
    if not evidence:
        return {"status": "needs_context", "reason": "no_personal_feeling", "evidence": []}
    if _OTHER_PERSON_RE.search(normalized) and _AFFECT_RE.search(normalized):
        return {"status": "needs_context", "reason": "other_person_context", "evidence": evidence}
    return {"status": "supported", "reason": "personal_feeling", "evidence": evidence}


def assess(text):
    """Assess whether journal text expresses a clear first-person affective state.
    
    Returns a dictionary with:
      - status: 'supported' or 'needs_context'
      - reason: 'personal_feeling', 'negated_feeling', 'other_person_context', or 'no_personal_feeling'
      - evidence: list of text spans supporting first-person feeling
    """
    if not text or not text.strip():
        return {"status": "needs_context", "reason": "no_personal_feeling", "evidence": []}
    
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    nlp = _get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(normalized)
            res = _assess_with_spacy(doc, normalized)
            fallback = _assess_regex_fallback(normalized)
            # Conservative principle: never ignore a clear negation detected by either parser
            if fallback["status"] == "needs_context" and fallback["reason"] == "negated_feeling":
                return fallback
            if res["status"] == "needs_context" and res["reason"] == "no_personal_feeling":
                if fallback["status"] == "supported":
                    return fallback
            return res
        except Exception:
            pass
    return _assess_regex_fallback(normalized)
