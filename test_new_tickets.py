"""Task 5 - Test the model with new tickets that were NOT used in training.

Run:  python src/test_new_tickets.py
Output: console table + outputs/new_ticket_test_results.csv
"""
import pandas as pd

from config import OUTPUT_DIR, RAW_CSV, ensure_dirs
from predict import predict_ticket
from preprocessing import clean_text

# (new ticket text, expected category) - written by hand, different wording from the dataset
TEST_CASES = [
    ("I can't remember my password and the site won't let me in", "Login Issue"),
    ("Getting an exception and the app closes whenever I press save", "Application Error"),
    ("Please send me the sales figures summary for October", "Report"),
    ("My phone number has changed, please update it in my profile", "Account Update"),
    ("Everything is crawling, pages are taking ages to open", "Performance"),
    ("I was debited twice for the same invoice", "Billing Issue"),
    ("The verification code never reaches my mobile when I try to sign in", "Login Issue"),
    ("System throws an unexpected error when uploading a file", "Application Error"),
    ("Need a report of overdue invoices by customer", "Report"),
    ("Please change the email id registered with my account", "Account Update"),
]


def main():
    ensure_dirs()
    # safety check: none of the test sentences may exist in the training data
    known = set()
    if RAW_CSV.exists():
        known = set(pd.read_csv(RAW_CSV)["ticket_description"].dropna().map(clean_text))
    overlap = [t for t, _ in TEST_CASES if clean_text(t) in known]
    assert not overlap, f"These test tickets are already in the dataset: {overlap}"

    rows = []
    for text, expected in TEST_CASES:
        r = predict_ticket(text)
        rows.append({
            "test_input": text,
            "expected_category": expected,
            "predicted_category": r["category"],
            "confidence_%": round(r["confidence"] * 100, 1),
            "result": "PASS" if r["category"] == expected else "FAIL",
        })
    res = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print("=" * 70)
    print("TASK 5 - NEW TICKET PREDICTION TESTS")
    print("=" * 70)
    print(res.to_string(index=False))
    passed = (res["result"] == "PASS").sum()
    print(f"\nCorrect: {passed} / {len(res)}  ({passed/len(res)*100:.0f}%)")
    res.to_csv(OUTPUT_DIR / "new_ticket_test_results.csv", index=False)


if __name__ == "__main__":
    main()
