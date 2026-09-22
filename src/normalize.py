"""
Brand name normalization utilities.

Goal: produce a canonical string for a brand name that is stable across
minor typographic / cosmetic variations (case, accents, punctuation,
legal suffixes, marketing noise) so exact-match can catch as many
retailer <-> Voysen pairs as possible before any fuzzy step runs.
"""
import re
import unicodedata

# Legal / marketing suffixes & noise tokens that carry no brand-identity
# signal. Applied only as whole trailing/leading tokens, never mid-string.
_NOISE_TOKENS = {
    "inc", "inc.", "ltd", "ltd.", "llc", "co", "co.", "corp", "corp.",
    "gmbh", "sa", "s.a", "spa", "s.p.a", "bv", "b.v", "srl", "s.r.l",
    "ag", "kg", "nv", "n.v", "plc", "sarl", "sas", "kk", "oy", "ab",
    "official", "store", "shop", "boutique", "brand", "brands",
}

_PAREN_RE = re.compile(r"[\(\)\[\]\{\}]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z&\s]")


def strip_accents(text: str) -> str:
    """Remove diacritics, e.g. 'Kérastase' -> 'Kerastase'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize(name: str) -> str:
    """
    Canonicalize a brand name for matching.

    Steps: unicode-fold -> strip accents -> lowercase -> drop
    parenthesised marketing text -> drop punctuation (keep '&') ->
    collapse whitespace -> drop leading/trailing legal-suffix tokens.
    Returns '' for null/empty/purely-symbolic input.
    """
    if name is None:
        return ""
    text = str(name)
    if not text.strip():
        return ""

    text = strip_accents(text)
    text = text.lower()
    text = _PAREN_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = text.split(" ")
    while tokens and tokens[-1] in _NOISE_TOKENS:
        tokens.pop()
    while tokens and tokens[0] in _NOISE_TOKENS:
        tokens.pop(0)

    return " ".join(tokens).strip()


def char_ngrams(text: str, n_min: int = 2, n_max: int = 4):
    """Yield character n-grams of a normalized, space-padded string.
    Used as the analyzer for the TF-IDF vectorizer (language-agnostic,
    works across Latin/Cyrillic/Japanese/etc. equally)."""
    padded = f" {text} "
    grams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            grams.append(padded[i : i + n])
    return grams
