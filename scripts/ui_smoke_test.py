"""Headlessly execute every Streamlit page via AppTest and report exceptions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent / "quant_intelligence"
# app.py is the navigation router; home_page.py is the Dashboard it opens by default.
pages = [ROOT / "app.py", ROOT / "home_page.py"] + sorted((ROOT / "pages").glob("*.py"))

failures = []
for page in pages:
    at = AppTest.from_file(str(page), default_timeout=60)
    try:
        at.run()
    except Exception as e:
        failures.append((page.name, f"harness exception: {e}"))
        continue
    if at.exception:
        failures.append((page.name, [str(e) for e in at.exception]))
    else:
        print(f"OK: {page.name} ({len(at.get('exception'))} exceptions, "
              f"{len(at.markdown)} markdown, {len(at.metric)} metrics rendered)")

print()
if failures:
    print("FAILURES:")
    for name, err in failures:
        print(f"  {name}: {err}")
    sys.exit(1)
else:
    print("ALL PAGES RENDERED WITHOUT EXCEPTIONS")
