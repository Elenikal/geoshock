"""
patch_event_detector.py  —  run from your geoshock_v2/ directory:

    python patch_event_detector.py

Fixes applied
─────────────
  Fix 1  GDELT URL: api.gdelt.org → api.gdeltproject.org  (465 Client Error)
  Fix 2  _clean_text: unicodedata + errors='ignore'        (UnicodeEncodeError)
  Fix 3  _clean_text calls moved inside try block          (unhandled exception)
"""

from pathlib import Path
import re

TARGET = Path("data/event_detector.py")
if not TARGET.exists():
    raise FileNotFoundError(
        f"Cannot find {TARGET}\n"
        "Make sure you run this script from your geoshock_v2/ directory."
    )

src = TARGET.read_text(encoding="utf-8")
original = src
fixes_applied = 0

# ── Fix 1: Wrong GDELT domain ────────────────────────────────────────────────
OLD_URL = '_GDELT_API = "https://api.gdelt.org/v2/doc/articles"'
NEW_URL = '_GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"'

if OLD_URL in src:
    src = src.replace(OLD_URL, NEW_URL)
    print("Fix 1 applied: GDELT URL -> api.gdeltproject.org/api/v2/doc/doc")
    fixes_applied += 1
elif NEW_URL in src:
    print("Fix 1 already applied - skipped")
else:
    print("WARNING Fix 1: could not find GDELT URL line - check manually")

# ── Fix 2: Replace _clean_text ───────────────────────────────────────────────
OLD_CLEAN = (
    'def _clean_text(text: str) -> str:\n'
    '    """Remove non-ASCII characters that break some API encodings."""\n'
    '    return text.encode("ascii", errors="replace").decode("ascii")'
)

NEW_CLEAN = (
    'def _clean_text(text: str) -> str:\n'
    '    """Normalise Unicode and strip non-ASCII. Never raises."""\n'
    '    import unicodedata\n'
    '    _MAP = {\n'
    '        "\\u2018": "\'",  "\\u2019": "\'",  # curly single quotes\n'
    '        "\\u201c": \'"\',  "\\u201d": \'"\',  # curly double quotes\n'
    '        "\\u2013": "-",   "\\u2014": "--",  # en/em dash\n'
    '        "\\u2026": "...", "\\u00b7": ".",   # ellipsis, middle dot\n'
    '        "\\u2032": "\'",  "\\u00b4": "\'",  # prime, acute\n'
    '    }\n'
    '    for uni, asc in _MAP.items():\n'
    '        text = text.replace(uni, asc)\n'
    '    text = unicodedata.normalize("NFKD", text)\n'
    '    return text.encode("ascii", errors="ignore").decode("ascii")'
)

if OLD_CLEAN in src:
    src = src.replace(OLD_CLEAN, NEW_CLEAN)
    print("Fix 2 applied: _clean_text uses unicodedata + errors='ignore'")
    fixes_applied += 1
elif "unicodedata.normalize" in src:
    print("Fix 2 already applied - skipped")
else:
    print("WARNING Fix 2: could not match _clean_text - check manually")

# ── Fix 3: Move clean calls inside try block ─────────────────────────────────
OLD_TRY = (
    '    # Clean unicode smart quotes and non-ASCII chars before sending\n'
    '    clean_headlines = [_clean_text(h) for h in headlines[:20]]\n'
    '    text = "\\n".join(f"- {h}" for h in clean_headlines)\n'
    '    system = _clean_text(_LLM_SYSTEM)\n'
    '    try:\n'
    '        client = anthropic.Anthropic(api_key=key)'
)

NEW_TRY = (
    '    try:\n'
    '        # Clean unicode smart quotes and non-ASCII chars before sending\n'
    '        clean_headlines = [_clean_text(h) for h in headlines[:20]]\n'
    '        text = "\\n".join(f"- {h}" for h in clean_headlines)\n'
    '        system = _clean_text(_LLM_SYSTEM)\n'
    '        client = anthropic.Anthropic(api_key=key)'
)

if OLD_TRY in src:
    src = src.replace(OLD_TRY, NEW_TRY)
    print("Fix 3 applied: cleaning calls moved inside try block")
    fixes_applied += 1
elif re.search(r'try:\s*\n\s*# Clean unicode', src):
    print("Fix 3 already applied - skipped")
else:
    print("WARNING Fix 3: could not match try-block pattern - check manually")

# ── Write ─────────────────────────────────────────────────────────────────────
if fixes_applied > 0:
    bak = TARGET.with_suffix(".py.bak")
    bak.write_text(original, encoding="utf-8")
    TARGET.write_text(src, encoding="utf-8")
    print(f"\n{fixes_applied} fix(es) written to {TARGET}")
    print(f"Original backed up to {bak}")
else:
    print("\nNo changes made.")

print("\nRestart: streamlit run dashboard/app.py")
