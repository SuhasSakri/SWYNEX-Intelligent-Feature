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
        # --- Billing (40) ---
        ("I was charged twice for my subscription this month", "Billing"),
        ("My invoice shows the wrong amount, please refund the difference", "Billing"),
        ("Can I get a receipt for last month's payment?", "Billing"),
        ("The card on file was declined but I was still charged", "Billing"),
        ("Why is my bill higher than usual this cycle?", "Billing"),
        ("I want to cancel my subscription and get a refund", "Billing"),
        ("Please update the billing address linked to my payment method", "Billing"),
        ("I never received an invoice for this month's charge", "Billing"),
        ("The upgrade fee seems incorrect, can you check?", "Billing"),
        ("My payment failed but the app still shows premium features locked", "Billing"),
        ("I was double billed for two different plans this cycle", "Billing"),
        ("Can I switch from monthly to annual billing?", "Billing"),
        ("The discount code I applied at checkout didn't reduce my invoice total", "Billing"),
        ("I need an itemized breakdown of last quarter's charges", "Billing"),
        ("How do I update the expiration date on the credit card used for billing?", "Billing"),
        ("I was charged after canceling my subscription last week", "Billing"),
        ("Is there a proration when I upgrade mid-cycle?", "Billing"),
        ("The sales tax on my invoice looks incorrect", "Billing"),
        ("Can you send me a copy of my payment history?", "Billing"),
        ("My free trial converted to a paid plan without warning and I was charged for it", "Billing"),
        ("I want to downgrade my plan to save money", "Billing"),
        ("The currency on my invoice is wrong, it should be in USD", "Billing"),
        ("Why was I charged for an add-on I never selected?", "Billing"),
        ("Please refund the annual plan I mistakenly purchased", "Billing"),
        ("My discount coupon expired before I could redeem it against an invoice", "Billing"),
        ("I need a copy of my billing statement for tax purposes", "Billing"),
        ("The renewal charge came through a day early", "Billing"),
        ("Can you waive the late payment fee shown on my invoice this one time?", "Billing"),
        ("I'm being billed for a seat I already removed", "Billing"),
        ("How do I add a purchase order number to my invoice?", "Billing"),
        ("The auto-renewal charged me despite canceling in the app", "Billing"),
        ("My payment method was rejected during checkout", "Billing"),
        ("Can I split my invoice into two payments?", "Billing"),
        ("I was charged the enterprise price but I'm on the starter plan", "Billing"),
        ("Please send an invoice addressed to my company, not me personally", "Billing"),
        ("The refund I was promised hasn't appeared on my card statement", "Billing"),
        ("Why does my invoice show a different plan name than what I signed up for?", "Billing"),
        ("I need to change who receives our monthly invoice emails", "Billing"),
        ("There's a mysterious charge on my card from your company", "Billing"),
        ("Can you prorate my refund since I'm cancelling mid-month?", "Billing"),

        # --- Technical (40) ---
        ("The app crashes every time I try to upload a file", "Technical"),
        ("I'm getting a 500 error when I try to log in", "Technical"),
        ("The dashboard is not loading any data for me", "Technical"),
        ("Export to PDF is broken, the file comes out empty", "Technical"),
        ("Push notifications stopped working after the last update", "Technical"),
        ("The search feature is broken and returns no results even for valid queries", "Technical"),
        ("I can't connect the app to my calendar, sync keeps failing", "Technical"),
        ("The mobile app freezes on the home screen", "Technical"),
        ("Our API integration keeps failing because requests are timing out constantly", "Technical"),
        ("Images are not rendering correctly in the report view, looks like a display bug", "Technical"),
        ("The site throws a 404 error on the settings page", "Technical"),
        ("Uploaded files disappear after a few minutes", "Technical"),
        ("The chart widget renders blank on Safari only", "Technical"),
        ("Webhooks stopped firing after yesterday's deploy", "Technical"),
        ("The app won't open at all since I updated my phone", "Technical"),
        ("I keep getting logged out every few minutes", "Technical"),
        ("The CSV export feature is broken and missing half the columns", "Technical"),
        ("Drag and drop stopped working in the editor", "Technical"),
        ("The app is extremely slow when loading large projects", "Technical"),
        ("I get a blank white screen after logging in", "Technical"),
        ("The integration with our CRM keeps throwing a connection error and disconnecting", "Technical"),
        ("Comments aren't saving when I click submit", "Technical"),
        ("Push notifications on mobile arrive hours late, seems like a delivery bug", "Technical"),
        ("Dark mode breaks the layout on the reports page", "Technical"),
        ("The app shows outdated data even after refreshing", "Technical"),
        ("I can't upload files larger than 5MB even though the limit says 50MB", "Technical"),
        ("The keyboard shortcuts stopped responding", "Technical"),
        ("Video playback buffers constantly in the app", "Technical"),
        ("My changes aren't syncing across devices", "Technical"),
        ("The app duplicates entries every time I save", "Technical"),
        ("Clicking the download button does nothing, the feature seems broken", "Technical"),
        ("The search bar throws a JavaScript error in the console", "Technical"),
        ("Two-way sync with Google Sheets is broken and data is out of date", "Technical"),
        ("The app crashes specifically when exporting large reports", "Technical"),
        ("I lose my draft every time the app refreshes automatically", "Technical"),
        ("The loading spinner never stops on the analytics page, the page seems stuck", "Technical"),
        ("The API returns a 429 rate limit error constantly", "Technical"),
        ("Copy-paste from Excel breaks the table formatting", "Technical"),
        ("The mobile app doesn't remember my login between sessions", "Technical"),
        ("Real-time collaboration cursors don't show up for teammates", "Technical"),

        # --- Account (40) ---
        ("I forgot my password and the reset email never arrives", "Account"),
        ("How do I change the email associated with my account?", "Account"),
        ("I need to add a teammate to my workspace", "Account"),
        ("My account got locked after too many login attempts", "Account"),
        ("Can you delete my account and all associated data?", "Account"),
        ("I want to transfer ownership of my workspace to a colleague", "Account"),
        ("Two-factor authentication isn't sending me a code when I try to log into my account", "Account"),
        ("I need to merge two accounts I accidentally created", "Account"),
        ("How do I change my username?", "Account"),
        ("My account shows the wrong company name, how do I fix it?", "Account"),
        ("I'm locked out of my account after changing my phone number", "Account"),
        ("Can you remove a former employee's account access?", "Account"),
        ("How do I downgrade my role from admin to member?", "Account"),
        ("My SSO login to my account keeps failing with an unknown error", "Account"),
        ("I can't invite new members to my account's organization", "Account"),
        ("How do I set up single sign-on for my company?", "Account"),
        ("My account was suspended and I don't know why", "Account"),
        ("I need to reset the admin password for our team account", "Account"),
        ("Can I have two people share admin access on one account?", "Account"),
        ("How do I revoke API keys tied to my account?", "Account"),
        ("My account profile photo won't update no matter how many times I try", "Account"),
        ("I accidentally deleted a teammate, can you restore their access?", "Account"),
        ("The verification link in my signup email has expired", "Account"),
        ("I want to change my account's primary contact email", "Account"),
        ("Can you audit who has access to our shared workspace?", "Account"),
        ("My login session keeps expiring after just a few minutes", "Account"),
        ("I need to recover an account I lost access to years ago", "Account"),
        ("How do I set custom permission levels for different team members?", "Account"),
        ("My account got flagged for suspicious activity and locked", "Account"),
        ("I can't find where to update my account's recovery phone number", "Account"),
        ("Please deactivate the account of an employee who just left", "Account"),
        ("How do I enable passkey login instead of a password?", "Account"),
        ("I need to change the organization name on our workspace", "Account"),
        ("My magic link login email never arrives", "Account"),
        ("Can you tell me who last logged into my account?", "Account"),
        ("I want to restrict login to only our company email domain", "Account"),
        ("How do I set up a backup admin in case I lose access?", "Account"),
        ("My account got merged with someone else's by mistake", "Account"),
        ("I need to update the security questions on my account", "Account"),
        ("Can you increase our workspace's seat limit?", "Account"),

        # --- General (40) ---
        ("What are your support hours?", "General"),
        ("Which platforms is your product available on besides the web?", "General"),
        ("Where can I find your general documentation and getting-started guides?", "General"),
        ("Is there a student discount available?", "General"),
        ("Can you tell me more about your offering for larger teams?", "General"),
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
        ("Is there somewhere you post updates about overall service reliability?", "General"),
        ("Is there a community forum I can join?", "General"),
        ("What's your data retention policy?", "General"),
        ("Do you offer a nonprofit discount?", "General"),
        ("How do I download your brand assets for a partner blog post?", "General"),
        ("Is there a certification program for your product?", "General"),
        ("Do you support Zapier integrations?", "General"),
        ("What's the best way to contact your sales team?", "General"),
        ("Do you have case studies from companies in retail?", "General"),
        ("Is your platform GDPR compliant?", "General"),
        ("What time zone are your support hours based on?", "General"),
        ("Can I get a demo environment to test before purchasing?", "General"),
        ("Do you have a partner or reseller program?", "General"),
        ("Where can I find your public API rate limits?", "General"),
        ("Is there a live chat option for quick questions?", "General"),
        ("Do you offer training webinars for new teams?", "General"),
        ("What's included in your onboarding package?", "General"),
        ("Can I request a feature that's not on your roadmap yet?", "General"),
        ("Do you publish uptime statistics anywhere?", "General"),
        ("Is there a marketplace for third-party plugins?", "General"),
        ("What happens to our data if we decide to stop using the product?", "General"),
        ("Do you have any compliance certifications you can share for a procurement review?", "General"),
        ("Can you share your product's public pricing page link?", "General"),
        ("Is there a trial or test environment we can experiment with before committing?", "General"),
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
