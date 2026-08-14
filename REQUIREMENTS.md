# AI Career Bootcamp — REQUIREMENTS (v1, 2026-08-13)

**Purpose of this document:** a complete, self-contained specification for building an
aggressive interview-prep bootcamp. It will be handed to a FRESH session to build —
that session will NOT have access to any prior context, so everything needed is in here.
**Hard rule: the bootcamp lives in `~/Desktop/AI-Career-Bootcamp/` and never touches the
`afghan-hr-monitor` repository or its Azure resources.**

---

## 1. Goal & outcome

Prepare the learner to pass **software-engineer / AI-engineer interviews at top-tier
companies (Apple, Google, OpenAI, Anthropic)** and, on the way, be immediately hireable
for applied-AI roles anywhere. Target readiness: **16 weeks, aggressive**. Success =
consistently solving LeetCode-medium problems in ≤25 min under observation, delivering
45-minute system-design answers at mid/senior level, and telling a compelling, deeply
technical story about a real production AI system.

## 2. The learner (calibrate everything to this)

- Career switcher: project-management background + master's-level data-science coursework
  (Python, ML basics, RL, NLP). Comfortable in Python; rusty on algorithmic problem-solving.
- Built and operates a real production AI platform end-to-end (see §6 — the capstone
  rebuilds it from spec).
- **Time budget:** drives rideshare ~8 h/day → that is AUDIO time. Desk time ~3–4 h/day.
  Every phase must ship **audio-ready material** (markdown files formatted for NotebookLM
  upload → Audio Overviews) alongside desk work.
- Budget-conscious: prefer free/audit options; flag anything paid with its price.

## 3. Program structure the builder must produce

- **16 weeks, 4 phases** (foundations → patterns → advanced + design → polish/mocks),
  with a printed **week-by-week schedule** and a **daily template**:
  - Desk (3–4 h): 2 timed coding problems + review; 3×/week a system-design session;
    capstone build blocks.
  - Car (audio): rotating queue — current phase's NotebookLM packs, design walkthroughs,
    out-loud rehearsal prompts (the builder writes the rehearsal scripts).
- **Weekly:** 1 mock interview (external platform or Claude-run), 1 spaced-repetition
  review day (redo failed problems), 1 rest half-day (sustainability is part of the spec).
- **Assessment gates** (the builder writes the actual tests):
  - Week 4: two LeetCode-easy in 40 min combined, no hints.
  - Week 8: one medium in 30 min + one 25-min mini system design.
  - Week 12: two mediums in 60 min + full 45-min design, mock conditions.
  - Week 16: full interview-day simulation (2 coding + 1 design + 1 behavioral).
- **Tracking:** a simple progress file (problems solved, times, mock scores, gate results).

## 4. Track A — Coding & algorithms (the core)

- Backbone: **NeetCode 150** (neetcode.io — free), pattern-ordered: arrays/hashing →
  two pointers → sliding window → stack → binary search → linked list → trees → tries →
  heaps → backtracking → graphs → intervals → greedy → 1-D DP → (2-D DP stretch).
- Method requirements: 25-min timer per problem; write the brute force first; study the
  optimal even when solved; **spaced repetition** (failed problems resurface at +3 days,
  +1 week, +1 month); all in Python; talk out loud while solving (interview conditioning).
- Books (priority order):
  1. *Grokking Algorithms* — Bhargava (week 1–2 warmup; friendly).
  2. *Cracking the Coding Interview* — McDowell (process + behavioral chapters early;
     problems as a supplement).
  3. *Elements of Programming Interviews in Python* — Aziz/Lee/Prakash (weeks 9+, the
     hard-mode supplement).
- Courses (pick per learner pace; all optional support for NeetCode):
  - Coursera: **Algorithms Part I & II** (Princeton/Sedgewick, free to audit) — theory
    backbone. Alternative: **Stanford Algorithms Specialization** (Roughgarden).
  - edX: **CS50** (Harvard) only if fundamentals feel shaky — likely skippable here.
  - Udemy (cheap on sale): **"Master the Coding Interview: DS & Algorithms"** (Neagoie)
    or **"Python for Data Structures, Algorithms and Interviews"** (Portilla).
  - MIT OCW **6.006** lecture videos (free) for depth on demand.
- The builder must produce: per-week problem lists, a solution-review checklist, and an
  **audio pack per pattern** (what the pattern is, when to reach for it, the 3 canonical
  problems narrated — for car listening).

## 5. Track B — System design (classic + ML flavor)

- Books: **System Design Interview vol 1 & 2** (Alex Xu) as the drill book;
  **Designing Data-Intensive Applications** (Kleppmann) as the depth book (has an
  audiobook — assign to car time); **Designing Machine Learning Systems** and
  **AI Engineering** (both Chip Huyen) for the ML-systems edge.
- Free video: ByteByteGo YouTube channel; jordanhasnolife / system-design-fight-club
  style walkthroughs optional.
- Optional paid: Educative's Grokking-style modern system-design course (flag price).
- Drill list the builder must schedule (one per session, answered OUT LOUD to a timer):
  URL shortener, rate limiter, KV store, news feed, chat, notification system, web
  crawler, YouTube, autocomplete, payment/idempotent webhook system, metrics pipeline —
  then the ML set: content-moderation pipeline, RAG system with grounded citations,
  LLM-serving gateway with caching/batching/fallbacks, evaluation platform for a
  classifier, near-duplicate detection at scale, feature store, recommendation feed.
- Requirement: every ML design session ends by mapping the answer back to the capstone
  system ("where does MY build do this, and what would I change at 100× scale?").
- Audio: the builder writes narrated design walkthroughs (problem → requirements →
  high-level → deep dives → tradeoffs) formatted for NotebookLM.

## 6. Track C — Capstone: rebuild "Sentinel" from spec (write every line yourself)

The learner's flagship interview story is a production human-rights-monitoring AI
platform. The bootcamp rebuilds an equivalent system — codename **Sentinel** — from this
spec, **from scratch, in a brand-new repo, no copying from the original codebase**. The
point: after 16 weeks the learner has personally typed every concept they'll be
interviewed on. Milestones (roughly weeks 3–14, interleaved ~1 build block/day):

1. **Ingestion** (wk 3–4): pluggable source adapters (RSS + one JSON API), content
   normalization, exact-hash dedup, SQLite via SQLAlchemy, typed models, pytest from day 1.
2. **Classifier cascade** (wk 5–6): cheap-model triage → stronger-model classification
   via a provider SDK with tool/structured output; confidence field; a golden-set eval
   harness computing per-class precision/recall; fail-open error handling; per-call
   token/cost logging table.
3. **Review queue** (wk 7): FastAPI + server-rendered Jinja admin; human confirm/reject
   with audit log; publish gate (nothing public without human confirmation).
4. **Near-duplicate detection** (wk 8): MinHash + LSH implemented BY HAND first (no
   library) against a small corpus, then swapped for `datasketch` — the learner must be
   able to whiteboard the algorithm afterward.
5. **RAG reports** (wk 9–10): retrieve from own DB → generate a weekly summary with
   per-claim citations → a grounding checker that verifies every number/ref exists in
   the retrieved context → human edit/approve/publish flow.
6. **Embeddings & retrieval** (wk 11): feature-hashing embedder from scratch (pure
   Python) + cosine similarity; then compare against a real embedding model; measure
   retrieval quality on a toy set.
7. **Hardening sprint** (wk 12–13): batch-API refactor of the cascade with sync
   fallback; prompt-injection defenses (delimited/JSON-wrapped untrusted content +
   policy prompt + suspicion flag → forced review); redaction layer with a
   contract test walking every public route; nightly backup job + a restore drill.
8. **Ship it** (wk 14): deploy somewhere free/cheap (Fly.io/Render/a $5 VPS —
   builder picks), README with architecture diagram, and a load test that FINDS the
   system's concurrency ceiling and documents it with the fix ladder.

Acceptance criteria per milestone = working code + tests + a 5-minute out-loud
explanation recorded as a rehearsal script. This track feeds Track F (each milestone →
a blog post candidate).

## 7. Track D — ML/LLM fundamentals (the "why" behind the capstone)

- **Coursera:** Machine Learning Specialization (Andrew Ng) — audit free; Deep Learning
  Specialization (audit) if time allows.
- **Free:** Karpathy's *Neural Networks: Zero to Hero* (YouTube — assign the
  micrograd/makemore/GPT videos); fast.ai *Practical Deep Learning* as an alternative
  path; DeepLearning.AI short courses on prompting/RAG/evals/agents (free, 1–2 h each —
  builder picks the 6 most relevant); Stanford CS229/CS224n lecture videos for depth
  on demand only.
- Emphasis: LLM-era practical knowledge — prompting, structured outputs, RAG design,
  evals, fine-tuning vs. RAG vs. prompting decisions, cost/latency engineering,
  safety/injection. The learner should be able to answer "how would you evaluate it /
  stop hallucinations / cut the cost 10×" fluently.

## 8. Track E — Behavioral & story

- Build a STAR story bank (8–10 stories) from the learner's real production system:
  hardest bug (a storage-layer locking bug across a network filesystem; an LLM that
  invented a summed figure caught by a grounding check), cost engineering (measured 50%
  API savings via batching + caching), safety-by-design (isolation architecture,
  redaction contract tests, PII gates), honest-limitations (a published figure that
  failed reproduction → public corrections log), constraint-driven architecture
  ($2k/yr, one maintainer).
- The 30-second / 2-minute / 10-minute project pitches, rehearsed as car audio.
- Company-specific angle sheets: Apple (privacy/craft), Google (scale/rigor),
  Anthropic/OpenAI (safety/evals/honest uncertainty).

## 9. Track F — Visibility (parallel, lightweight)

- 3 technical blog posts over the 16 weeks (candidates: "a grounding check that caught
  my LLM doing arithmetic", "cutting LLM costs 50% with batch APIs — with real numbers",
  "MinHash-LSH explained by building it"). Platform: personal blog or Medium/dev.to.
- Open-source one clean extraction from Sentinel (e.g., the grounding checker or the
  batch-with-fallback pattern) as a small documented library.
- LinkedIn/GitHub hygiene pass in week 2 and week 15.

## 10. Mock interviews

- External: Pramp (free) weekly from week 4; interviewing.io (paid — flag) for 2–3
  sessions in weeks 12–16.
- Claude-run mocks (the building session should create prompt scripts for these): timed
  coding with an interviewer persona that probes complexity and edge cases; 45-min
  design interviews with follow-up grilling; behavioral with scoring rubrics. At least
  one full "interview day" simulation in week 16.

## 11. THE DASHBOARD — the bootcamp IS a website (primary deliverable)

The learner experiences the whole bootcamp through a **clean local website**: open it,
see **today**, do the tasks, check them off, done. Requirements:

**Core experience:**
- **Today view (the home screen):** today's date, phase + week + day number, and the
  day's task list — each task with a checkbox, a time estimate, and expandable detail
  (the actual problem links / reading pages / build-milestone brief / audio queue).
  Checking the last task marks the day complete with a satisfying "day done" state.
- **Task types rendered distinctly:** coding problems (with links to the exact NeetCode/
  LeetCode problem + a built-in **25-minute countdown timer**), reading (book + pages),
  video (link + duration), system-design drill (the prompt + a 45-min timer), Sentinel
  build block (milestone brief + acceptance checklist), car-audio queue (what to load
  into NotebookLM / listen to today), rehearsal prompts (say-out-loud scripts).
- **Progress everywhere:** phase progress bars, week strip (done/today/upcoming),
  problems-solved counter by pattern, streak counter, and the four **assessment gates**
  rendered as special "boss day" pages with pass/fail recording.
- **Spaced repetition built in:** a failed coding task automatically reappears in the
  Today view at +3 days / +1 week / +1 month (mark tasks pass/fail, not just done).
- **Missed-day behavior:** the schedule anchors to a **start date chosen on first
  launch**, never silently skips — missed days roll into the built-in catch-up slack
  (every 4th week) and the Today view says exactly what to do about it.

**Technical requirements (keep it boring and unbreakable):**
- Runs **fully offline and locally**: either a pure static site (`index.html` +
  `curriculum.json`, double-click to open) or a one-command tiny local server —
  builder's choice, but zero cloud dependencies, zero accounts, zero build tooling
  beyond what a fresh laptop has.
- **Mobile-friendly first** — the learner checks it on a phone between rides; the Today
  view must be perfectly usable one-handed on a phone screen.
- Progress must persist AND be inspectable/portable as a plain file (JSON or MD) — a
  future Claude session should be able to read progress and adjust the plan. Include
  export/import so progress survives a browser change.
- All curriculum content lives in **data files** (e.g., `curriculum.json`), NOT
  hardcoded in the HTML — so weeks can be edited without touching the app.
- Clean, calm visual design: readable typography, generous spacing, light+dark themes,
  no clutter — the page should make you want to start, not overwhelm you.

**Supporting artifacts (linked from the dashboard, also usable standalone):**
1. `SCHEDULE.md` — the same 16 weeks in plain markdown (the dashboard's source of truth,
   human-readable fallback).
2. `tracks/` — per-track weekly assignments and checklists.
3. `audio/` — NotebookLM-ready packs (pattern narrations, design walkthroughs, rehearsal
   scripts, story bank) — first-class deliverable; the dashboard's daily audio queue
   points at these files.
4. `sentinel-spec/` — the capstone milestone briefs with acceptance criteria (§6 expanded).
5. `mocks/` — Claude-mock-interview scripts + scoring rubrics.
6. `progress.json` (+ human-readable mirror) — the tracking store with gate results.
7. `README.md` — how to launch the dashboard and start week 1, day 1.

## 12. Non-goals / hard rules

- No research-scientist track (no papers) — applied engineering roles only.
- Do NOT touch, copy from, or reference the `afghan-hr-monitor` repo, its data, or its
  Azure resources. Sentinel is built clean-room from §6's spec only.
- No course-buying sprees: default to free/audit; every paid item needs a stated price
  and a reason a free option doesn't cover it.
- Sustainable aggression: 6 days/week max; the schedule must survive a bad week
  (built-in catch-up slack every 4th week).
