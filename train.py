"""Task 3 + Task 4 - Build, train and evaluate the ticket classifier.

Pipeline:  cleaned text -> TF-IDF (1-2 word n-grams) -> Logistic Regression

Run:  python src/train.py
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (CLEAN_CSV, MODEL_INFO_PATH, MODEL_PATH, OUTPUT_DIR, RANDOM_STATE,
                    SCREENSHOT_DIR, TEST_SIZE, ensure_dirs)


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def build_final_model():
    """The model we ship: TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", LogisticRegression(C=10, max_iter=1000)),
    ])


def compare_models(X_train, y_train):
    """5-fold cross-validation on the TRAINING data only (test set stays untouched)."""
    candidates = {
        "Logistic Regression": LogisticRegression(C=10, max_iter=1000),
        "Naive Bayes (Multinomial)": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        rows.append({"model": name, "cv_f1_macro_mean": scores.mean(), "cv_f1_macro_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)


def save_confusion_matrix(cm, labels):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (counts)")
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap="Greens", xticklabels=labels, yticklabels=labels,
                ax=axes[1], cbar=False)
    axes[1].set_title("Confusion Matrix (% of each true category)")
    for ax in axes:
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("Actual category")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "confusion_matrix.png", dpi=110)
    plt.close(fig)


def main(quiet=False):
    ensure_dirs()
    say = (lambda *a, **k: None) if quiet else print

    if not CLEAN_CSV.exists():
        import preprocessing
        preprocessing.main()
    df = pd.read_csv(CLEAN_CSV)
    X, y = df["clean_description"], df["category"]

    say("=" * 70)
    say("TASK 3 & 4 - MODEL TRAINING AND TESTING")
    say("=" * 70)

    # ---- split (stratified: every category keeps the same share in both parts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    say(f"\nTraining tickets: {len(X_train)} ({100*(1-TEST_SIZE):.0f}%)   "
        f"Testing tickets: {len(X_test)} ({100*TEST_SIZE:.0f}%)")

    # ---- compare candidate algorithms (cross-validation on training data)
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    say("\nAlgorithm comparison (5-fold CV, macro F1, training data only):")
    say(comparison.round(4).to_string(index=False))

    # ---- train final model
    model = build_final_model()
    model.fit(X_train, y_train)

    # ---- evaluate on the unseen test data
    y_pred = model.predict(X_test)
    labels = sorted(y.unique())
    metrics = {
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "labels": labels,
    }
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    report_txt = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
    metrics["per_class"] = {l: {k: float(report_dict[l][k]) for k in ("precision", "recall", "f1-score", "support")}
                            for l in labels}

    say("\n----- RESULTS ON TEST DATA -----")
    say(f"Accuracy           : {metrics['accuracy']*100:.2f}%")
    say(f"Precision (weighted): {metrics['precision_weighted']*100:.2f}%")
    say(f"Recall    (weighted): {metrics['recall_weighted']*100:.2f}%")
    say(f"F1 score  (weighted): {metrics['f1_weighted']*100:.2f}%")
    say(f"(Training accuracy  : {metrics['train_accuracy']*100:.2f}%)")
    say("\nPer-category report:\n" + report_txt)
    say("Confusion matrix (rows = actual, columns = predicted):")
    say(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    errors = pd.DataFrame({"description": X_test.values, "actual": y_test.values, "predicted": y_pred})
    errors = errors[errors["actual"] != errors["predicted"]]
    say(f"\nMisclassified test tickets: {len(errors)}")
    if len(errors):
        say(errors.to_string(index=False))
    errors.to_csv(OUTPUT_DIR / "misclassified_test_tickets.csv", index=False)

    # ---- which words drive each category? (explainability)
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = np.array(tfidf.get_feature_names_out())
    top_features = {}
    for i, cls in enumerate(clf.classes_):
        top_features[cls] = names[np.argsort(clf.coef_[i])[::-1][:6]].tolist()
    metrics["top_features"] = top_features
    say("\nWords/phrases with the strongest influence per category:")
    for cls, feats in top_features.items():
        say(f"  {cls:<18}: {', '.join(feats)}")

    # ---- save everything
    save_confusion_matrix(cm, labels)
    joblib.dump(model, MODEL_PATH)
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({"sklearn_version": sklearn.__version__}, f)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
    say(f"\nModel saved -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()
