# Task 3 — Intelligent Feature: Priority, SLA & Human-Review Routing

Builds on Tasks 1 (design) and 2 (baseline classifier) for the Support Ticket Triage
Classifier. This task adds a reasoning layer on top of the raw prediction, rigorous
evaluation, real failure-case analysis, and defensive error handling.

## What's new here

1. **The intelligent feature** — `classify_ticket()` in `ticket_intelligence.py` doesn't
   just return a category. It layers on:
   - **Priority** (High/Medium/Low), escalated automatically if urgency language ("urgent",
     "asap", "locked out", etc.) appears in the ticket
   - **SLA hours**, derived from priority
   - **Routing team** suggestion
   - **`needs_human_review`** — `True` whenever the model's confidence is below 0.40, so
     low-confidence predictions get a human in the loop instead of being auto-actioned
   - **An auto-generated first-response draft**

2. **More robust evaluation** — added **5-fold cross-validation** alongside the Task 2
   train/test split, so the reported metric reflects performance across the whole dataset
   rather than a single split.

3. **Failure-case analysis** — a handful of concrete misclassifications pulled from the
   held-out predictions, each with a root-cause explanation (see `demo.ipynb`, section 2).
   This points directly at the next improvement for Task 4: more labeled examples per
   category so similar-sounding tickets separate more cleanly.

4. **Error handling** — `classify_ticket()` never raises. Empty strings, `None`, wrong
   types, too-short text, and oversized text are all caught and returned as structured
   results instead of exceptions.

## How to run

```bash
pip install -r requirements.txt
python ticket_intelligence.py       # trains, cross-validates, and demos error handling
jupyter notebook demo.ipynb         # full walkthrough with evaluation + failure cases
```

## Files
- `ticket_intelligence.py` — training, cross-validation, and the `classify_ticket()`
  intelligent feature with error handling
- `demo.ipynb` — executed notebook: evaluation, failure-case analysis, error-handling
  tests, and intelligent-feature demo, in one place
- `requirements.txt` — dependencies (scikit-learn, numpy, jupyter, nbformat, nbclient)
