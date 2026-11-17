"""Task 5 + Task 6 (Option A) - Predict the category of a NEW ticket.

Interactive command-line app:
    python src/predict.py

One-shot prediction:
    python src/predict.py --text "I forgot my password and cannot login"
"""
import argparse
import json

import joblib
import sklearn

from config import LOW_CONFIDENCE, MODEL_INFO_PATH, MODEL_PATH
from preprocessing import clean_text
from responses import get_response

_MODEL = None


def _model_is_compatible():
    """The saved model must have been created with the same scikit-learn version."""
    try:
        with open(MODEL_INFO_PATH, encoding="utf-8") as f:
            return json.load(f).get("sklearn_version") == sklearn.__version__
    except (OSError, ValueError):
        return False


def load_model():
    """Load the trained model. If it is missing/incompatible, train it automatically."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not MODEL_PATH.exists() or not _model_is_compatible():
        print("[info] Trained model not found or built with another scikit-learn version "
              "- training it now (takes a few seconds)...")
        import train
        train.main(quiet=True)
    _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def predict_ticket(text, top_n=3):
    """Return a dict with category, confidence (0-1), top-N probabilities and a flag."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Ticket description is empty.")
    model = load_model()
    proba = model.predict_proba([cleaned])[0]
    order = proba.argsort()[::-1]
    classes = model.classes_
    category = str(classes[order[0]])
    confidence = float(proba[order[0]])
    return {
        "category": category,
        "confidence": confidence,
        "top": [(str(classes[i]), float(proba[i])) for i in order[:top_n]],
        "needs_review": confidence < LOW_CONFIDENCE,
        "suggested_response": get_response(category),
    }


def format_result(result):
    lines = [
        "Predicted Category:",
        f"  {result['category']}",
        f"Confidence: {result['confidence']*100:.1f}%",
    ]
    if result["needs_review"]:
        lines.append("Note: low confidence - please let a human agent review this ticket.")
    lines += ["", "Suggested Response:", f"  {result['suggested_response']}"]
    return "\n".join(lines)


def run_cli():
    print("=" * 36)
    print(" Customer Support Classifier")
    print("=" * 36)
    print("Type a ticket description and press Enter. Type 'q' to quit.\n")
    while True:
        try:
            text = input("Enter Ticket Description:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break
        if not text:
            print("Please type a description.\n")
            continue
        try:
            print("\n" + format_result(predict_ticket(text)) + "\n")
        except ValueError as e:
            print(f"{e}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict the category of a support ticket.")
    parser.add_argument("--text", help="ticket description (skips interactive mode)")
    args = parser.parse_args()
    if args.text:
        try:
            print(format_result(predict_ticket(args.text)))
        except ValueError as e:
            print(e)
    else:
        run_cli()


if __name__ == "__main__":
    main()
