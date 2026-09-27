"""
ticket_intelligence.py
Task 3: Intelligent Feature (SWYNEX AI Internship)

Builds on the Task 2 classifier (TF-IDF + Logistic Regression) and adds an
intelligent decision layer on top of the raw prediction:

  1. Priority assignment (High / Medium / Low) based on category + urgency cues
  2. SLA countdown derived from priority
  3. Routing-team suggestion
  4. Confidence-gated "needs_human_review" flag — low-confidence predictions
     are routed to a human instead of being auto-actioned
  5. An auto-generated first-response draft

All of this is wrapped in defensive error handling so bad input (empty text,
wrong type, gibberish, extremely long input) degrades gracefully instead of
crashing the pipeline.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

MAX_TICKET_LENGTH = 2000          # characters; longer input is truncated
MIN_TICKET_LENGTH = 3             # characters; shorter input is rejected
CONFIDENCE_REVIEW_THRESHOLD = 0.40  # below this, flag for human review

ROUTING_TEAM = {
    "Billing": "Finance & Billing Ops",
    "Technical": "Platform Engineering",
    "Account": "Identity & Access Support",
    "General": "Customer Success",
}

BASE_PRIORITY = {
    "Billing": "Medium",
    "Technical": "High",
    "Account": "Medium",
    "General": "Low",
}

SLA_HOURS = {"High": 4, "Medium": 24, "Low": 72}

URGENCY_KEYWORDS = ["urgent", "asap", "immediately", "can't access", "cannot access",
                    "locked out", "down", "not working at all", "critical"]

RESPONSE_TEMPLATES = {
    "Billing": "Hi, thanks for reaching out about your billing concern. We're looking into "
               "the charge/invoice you mentioned and will follow up with a resolution shortly.",
    "Technical": "Hi, sorry for the trouble you're running into. Our engineering team has been "
                 "notified and is investigating the issue you described.",
    "Account": "Hi, thanks for letting us know about your account issue. We're verifying your "
               "details and will help restore access as quickly as possible.",
    "General": "Hi, thanks for your question! We'll get back to you shortly with the "
               "information you're looking for.",
}


# ---------------------------------------------------------------------------
# Training (same dataset/approach as Task 2, isolated into a function)
# ---------------------------------------------------------------------------
def get_training_data():
    data = [
        ("I was charged twice for my subscription this month", "Billing"),
        ("My invoice shows the wrong amount, please refund the difference", "Billing"),
        ("Can I get a receipt for last month's payment?", "Billing"),
        ("The card on file was declined but I was still charged", "Billing"),
        ("Why is my bill higher than usual this cycle?", "Billing"),
        ("I want to cancel my subscription and get a refund", "Billing"),
        ("Please update my billing address on the account", "Billing"),
        ("I never received an invoice for this month's charge", "Billing"),
        ("The upgrade fee seems incorrect, can you check?", "Billing"),
        ("My payment failed but the app still shows premium features locked", "Billing"),
        ("I was double billed for two different plans this cycle", "Billing"),
        ("Can I switch from monthly to annual billing?", "Billing"),
        ("The discount code I applied didn't reduce my total", "Billing"),
        ("I need an itemized breakdown of last quarter's charges", "Billing"),

        ("The app crashes every time I try to upload a file", "Technical"),
        ("I'm getting a 500 error when I try to log in", "Technical"),
        ("The dashboard is not loading any data for me", "Technical"),
        ("Export to PDF is broken, the file comes out empty", "Technical"),
        ("Push notifications stopped working after the last update", "Technical"),
        ("The search feature returns no results even for valid queries", "Technical"),
        ("I can't connect the app to my calendar, sync keeps failing", "Technical"),
        ("The mobile app freezes on the home screen", "Technical"),
        ("API requests are timing out constantly today", "Technical"),
        ("Images are not rendering correctly in the report view", "Technical"),
        ("The site throws a 404 error on the settings page", "Technical"),
        ("Uploaded files disappear after a few minutes", "Technical"),
        ("The chart widget renders blank on Safari only", "Technical"),
        ("Webhooks stopped firing after yesterday's deploy", "Technical"),

        ("I forgot my password and the reset email never arrives", "Account"),
        ("How do I change the email associated with my account?", "Account"),
        ("I need to add a teammate to my workspace", "Account"),
        ("My account got locked after too many login attempts", "Account"),
        ("Can you delete my account and all associated data?", "Account"),
        ("I want to transfer ownership of my workspace to a colleague", "Account"),
        ("Two-factor authentication isn't sending me a code", "Account"),
        ("I need to merge two accounts I accidentally created", "Account"),
        ("How do I change my username?", "Account"),
        ("My account shows the wrong company name, how do I fix it?", "Account"),
        ("I'm locked out after changing my phone number", "Account"),
        ("Can you remove a former employee's account access?", "Account"),
        ("How do I downgrade my role from admin to member?", "Account"),
        ("My SSO login keeps failing with an unknown error", "Account"),

        ("What are your support hours?", "General"),
        ("Do you have a mobile app for iOS?", "General"),
        ("Where can I find your API documentation?", "General"),
        ("Is there a student discount available?", "General"),
        ("Can you tell me more about your enterprise plan?", "General"),
        ("How do I get started with your product?", "General"),
        ("Do you offer onboarding calls for new customers?", "General"),
        ("What integrations do you support?", "General"),
        ("Is there a public roadmap I can follow?", "General"),
        ("How can I give feedback on a feature request?", "General"),
        ("Do you have a referral or affiliate program?", "General"),
        ("What's the difference between the free and paid plans?", "General"),
        ("Where can I read your terms of service?", "General"),
        ("Do you have a changelog for recent releases?", "General"),
        ("Can I schedule a product demo with your sales team?", "General"),
        ("What languages does your platform support?", "General"),
    ]
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    return texts, labels


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def cross_validate_model(n_splits=5, random_state=42):
    """
    A single train/test split on 58 examples is noisy (macro F1 swings widely
    with the random seed). Stratified k-fold cross-validation is the more
    honest way to report a single-number metric on a dataset this small: it
    averages performance across every example instead of trusting one split.
    """
    texts, labels = get_training_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(build_pipeline(), texts, labels, cv=skf, scoring="f1_macro")
    return {"fold_scores": scores, "mean_macro_f1": scores.mean(), "std_macro_f1": scores.std()}


def train_model(test_size=0.25, random_state=42):
    """
    Fits on a single train/test split so we have concrete held-out
    predictions to inspect (used for the per-class report and the failure
    case analysis). See cross_validate_model() for the more robust overall
    metric used as the headline number.
    """
    texts, labels = get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    eval_report = {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report_text": classification_report(y_test, y_pred, zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }
    return model, eval_report


def train_final_model():
    """Fits on the full labeled dataset — the model actually used to serve predictions."""
    texts, labels = get_training_data()
    model = build_pipeline()
    model.fit(texts, labels)
    return model


# ---------------------------------------------------------------------------
# The intelligent feature: classify + reason about the result, safely
# ---------------------------------------------------------------------------
def classify_ticket(model, text, confidence_threshold=CONFIDENCE_REVIEW_THRESHOLD):
    """
    Classifies a ticket and layers priority/SLA/routing/auto-response logic
    on top. Never raises — always returns a dict, with an "error" key set
    if the input couldn't be processed.
    """
    # --- Error handling: type validation ---
    if text is None:
        return {"error": "No ticket text was provided (received None)."}
    if not isinstance(text, str):
        return {"error": f"Ticket text must be a string, got {type(text).__name__}."}

    cleaned = text.strip()

    # --- Error handling: empty / whitespace-only input ---
    if len(cleaned) == 0:
        return {"error": "Ticket text is empty. Please provide the ticket's content."}

    # --- Error handling: too short to classify meaningfully ---
    if len(cleaned) < MIN_TICKET_LENGTH:
        return {"error": f"Ticket text is too short ({len(cleaned)} chars) to classify reliably."}

    # --- Error handling: excessively long input gets truncated, not rejected ---
    truncated = False
    if len(cleaned) > MAX_TICKET_LENGTH:
        cleaned = cleaned[:MAX_TICKET_LENGTH]
        truncated = True

    # --- Model inference, defensively wrapped ---
    try:
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        pred_idx = probs.argmax()
        category = classes[pred_idx]
        confidence = float(probs[pred_idx])
    except Exception as exc:  # model/vectorizer failure, unexpected input, etc.
        return {"error": f"Classification failed unexpectedly: {exc}"}

    # --- Reasoning layer on top of the raw prediction ---
    needs_human_review = confidence < confidence_threshold

    priority = BASE_PRIORITY.get(category, "Medium")
    lowered = cleaned.lower()
    if any(kw in lowered for kw in URGENCY_KEYWORDS) and priority != "High":
        priority = "High"  # escalate on urgency language regardless of category

    result = {
        "text": cleaned,
        "truncated": truncated,
        "category": category,
        "confidence": round(confidence, 4),
        "needs_human_review": needs_human_review,
        "priority": priority,
        "sla_hours": SLA_HOURS[priority],
        "routing_team": ROUTING_TEAM.get(category, "General Support"),
        "auto_response_draft": RESPONSE_TEMPLATES.get(category, "Thanks, we'll get back to you shortly."),
    }
    return result


if __name__ == "__main__":
    cv = cross_validate_model()
    print(f"5-fold CV macro F1: {cv['mean_macro_f1']:.3f} (+/- {cv['std_macro_f1']:.3f})")
    print("Per-fold scores:", np.round(cv["fold_scores"], 3))

    model, eval_report = train_model()
    print("\nHeld-out split macro F1:", round(eval_report["macro_f1"], 3))
    print(eval_report["report_text"])

    print("\n--- classify_ticket() demo ---")
    for t in [
        "The app keeps crashing, this is urgent and I can't work",
        "",
        None,
        12345,
        "hi",
        "Do you offer a referral program?",
    ]:
        print(t, "->", classify_ticket(model, t))
