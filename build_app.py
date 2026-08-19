"""Inject curriculum.json into app_template.html -> index.html (single self-contained file)."""
import json, pathlib

here = pathlib.Path(__file__).parent
tpl = (here / "app_template.html").read_text()
cur = json.loads((here / "curriculum.json").read_text())
exs = json.loads((here / "exercises.json").read_text())

def blob(o):  # compact, with a </script> guard
    return json.dumps(o, separators=(",", ":")).replace("</", "<\\/")

out = tpl.replace("__CURRICULUM__", blob(cur)).replace("__EXERCISES__", blob(exs))

assert "__CURRICULUM__" not in out and "__EXERCISES__" not in out, "placeholder left behind"
(here / "index.html").write_text(out)
kb = len(out) / 1024
print(f"index.html written — {kb:.0f} KB, {len(cur['days'])} days, "
      f"{sum(len(d['tasks']) for d in cur['days'])} tasks, "
      f"{len(exs)} exercises "
      f"({sum(e['tests'].count('check(') for e in exs.values())} checks)")
