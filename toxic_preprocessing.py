"""Shared text cleaning for the toxic-comment project.
Use the SAME functions for training, evaluation and the Streamlit app."""
import re, html, codecs, unicodedata

# ---- mojibake repair: UTF-8 text that was decoded as Windows-1252, e.g. "I\u00e2\u20ac\u2122ll" -> "I'll" ----
_MOJIBAKE_HINT = re.compile("\u00c3|\u00c2|\u00e2\u20ac|\u00e2\u201a|\u00ef\u00bc|\u00e3\u0192|\u00cf\u2030|\u00f0\u0178")

def _c1_passthrough(exc):
    """Let cp1252 encoding pass through the 5 control chars it cannot represent (0x81, 0x8d, 0x8f, 0x90, 0x9d)."""
    if isinstance(exc, UnicodeEncodeError):
        bad = exc.object[exc.start:exc.end]
        if all(0x80 <= ord(c) <= 0x9F for c in bad):
            return bytes(ord(c) for c in bad), exc.end
    raise exc
codecs.register_error("c1passthrough", _c1_passthrough)

_MOJIBAKE_FALLBACK = {            # used when a string mixes mojibake with genuine non-cp1252 characters
    "\u00e2\u20ac\u2122": "'", "\u00e2\u20ac\u02dc": "'", "\u00e2\u20ac\u0153": '"', "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u00a6": "...", "\u00e2\u20ac\u201d": " - ", "\u00e2\u20ac\u201c": " - ",
}

def fix_mojibake(text):
    if not _MOJIBAKE_HINT.search(text):
        return text
    try:
        return text.encode("cp1252", errors="c1passthrough").decode("utf-8")
    except UnicodeError:
        for bad, good in _MOJIBAKE_FALLBACK.items():
            text = text.replace(bad, good)
        return text

# ---- patterns ----
_URL_RE      = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")            # only real tags; keeps "<3" and "a < b"
_MENTION_RE  = re.compile(r"(?<!\w)@\w+")
_ZEROWIDTH   = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL_RE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_REPEAT_RE   = re.compile(r"(.)\1{3,}")                    # 4+ identical chars -> 3
_SPACE_RE    = re.compile(r"\s+")
_PUNCT_MAP   = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
                              "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2013": " - ", "\u2014": " - ", "\u2212": "-"})

# used by the notebook to report how many comments each rule affects
PATTERNS = {
    "mojibake (encoding damage)": _MOJIBAKE_HINT,
    "HTML entities (&amp; ...)": re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);"),
    "HTML tags": _HTML_TAG_RE,
    "URLs": _URL_RE,
    "@mentions": _MENTION_RE,
    "repeated characters (4+)": _REPEAT_RE,
    "line breaks / tabs": re.compile(r"[\r\n\t]"),
    "curly quotes / long dashes": re.compile("[\u2018\u2019\u201c\u201d\u2013\u2014]"),
    "zero-width / control chars": re.compile(_ZEROWIDTH.pattern + "|" + _CONTROL_RE.pattern),
}

def clean_text(text):
    """Light cleaning that preserves casing, punctuation and identity terms."""
    if not isinstance(text, str):
        return ""
    t = fix_mojibake(text)
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(_PUNCT_MAP)
    t = _ZEROWIDTH.sub("", t)
    t = _CONTROL_RE.sub(" ", t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = _URL_RE.sub(" [URL] ", t)
    t = _MENTION_RE.sub("@user", t)
    t = _REPEAT_RE.sub(r"\1\1\1", t)
    return _SPACE_RE.sub(" ", t).strip()

def clean_for_tfidf(text):
    """clean_text + lowercase, used for TF-IDF style models."""
    return clean_text(text).lower()

def has_content(text):
    """False if nothing is left once the placeholders are removed."""
    return bool(re.search(r"\w", text.replace("[URL]", "").replace("@user", "")))
