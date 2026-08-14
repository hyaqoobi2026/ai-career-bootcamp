"""Inject curriculum.json into app_template.html -> index.html (single self-contained file)."""
import json, pathlib

here = pathlib.Path(__file__).parent
tpl = (here / "app_template.html").read_text()
cur = json.loads((here / "curriculum.json").read_text())

# compact JSON keeps the page small; </script> guard for safety
blob = json.dumps(cur, separators=(",", ":")).replace("</", "<\\/")
out = tpl.replace("__CURRICULUM__", blob)

(here / "index.html").write_text(out)
kb = len(out) / 1024
print(f"index.html written — {kb:.0f} KB, {len(cur['days'])} days, "
      f"{sum(len(d['tasks']) for d in cur['days'])} tasks")
