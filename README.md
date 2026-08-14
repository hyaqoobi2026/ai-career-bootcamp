# AI Career Bootcamp

A 16-week, day-by-day training plan — **Sun 30 Aug 2026 → Sat 19 Dec 2026** — as a
self-contained web app you open on your phone each morning.

**112 days · 466 tasks · ~42 focused hrs/week + ~35 audio hrs/week while driving**

## Using it

Open `index.html`. It shows today's tasks, grouped into blocks:

| Block | When | What |
|---|---|---|
| Morning — deep work | before driving, 3h | implementation + math, from scratch |
| Evening — build & learn | after driving, 3h | project work, courses, reading |
| Job hunt | 1.5h | applications, follow-ups, networking |
| While driving | ~5h | audio — podcasts, lectures, your own NotebookLM packs |

Tap a task to see exactly what to do. Tap the box to check it off. Progress saves in
the browser on that device — use **Export** to move it between devices.

### On your phone

Once it is served over HTTPS (see below), open it in Safari → Share → **Add to Home
Screen**. It then behaves like a native app, and works offline after first load.

## The 40 focused hours

| Activity | hrs/wk | share |
|---|---|---|
| Implementation (code from scratch) | 12 | 30% |
| Project work (AHRC as the lab) | 8 | 20% |
| Math (worked problems + code) | 6 | 15% |
| Guided courses (Karpathy, fast.ai) | 6 | 15% |
| Reading + notes | 4 | 10% |
| Flex / catch-up | 4 | 10% |

Job hunt is a separate 10 hrs/week and takes priority over everything here — the plan
is worth nothing if it delays employment.

## The four phases

1. **Foundations** (wks 1–4) — linear/logistic regression, backprop, micrograd, classical ML
2. **Deep Learning Core** (wks 5–8) — PyTorch, CNNs, makemore, attention from scratch
3. **Transformers & LLMs** (wks 9–12) — nanoGPT, your own GPT on Dari/Pashto, fine-tuning, evaluation
4. **The Capstone Experiment** (wks 13–16) — fine-tune on your ~477 human-reviewed
   incidents, benchmark against the 3-tier LLM cascade on accuracy/latency/cost, write it up, ship it

Assessment days ("gates") close each phase: **26 Sep, 24 Oct, 21 Nov, 19 Dec** — no
references, no help. A bad result is information, not failure.

## Editing the plan

The curriculum is data, not markup. To change it:

```bash
python3 build_curriculum.py   # edit the W{} / AUDIO{} tables first -> curriculum.json
python3 build_app.py          # inlines the JSON into index.html
```

`index.html` is fully self-contained — no network requests, no dependencies, no build
step for the reader. Open the file and it runs.

## Publishing it (needed for phone access)

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
gh api -X POST repos/:owner/:repo/pages -f 'source[branch]=main' -f 'source[path]=/'
```

Live a minute later at `https://<user>.github.io/ai-career-bootcamp/`.
