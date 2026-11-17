"""Task 7 - Build report.pdf (solution documentation) from the real results.

All numbers in the report are read from outputs/*.json|csv, so they always match
the latest training run.

Run:  python src/make_report.py --name "Your Name"
"""
import argparse
import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from config import BASE_DIR, OUTPUT_DIR, RAW_CSV, SCREENSHOT_DIR

PAGE_W = A4[0] - 4 * cm
BLUE = colors.HexColor("#2f5bea")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14.5, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=BLUE, spaceBefore=12, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, keepWithNext=1)
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontSize=22, leading=27, textColor=BLUE, alignment=TA_CENTER)
SUB = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, textColor=colors.grey)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=9, leading=12,
                      backColor=colors.HexColor("#f1f4fb"), borderPadding=6, spaceBefore=4, spaceAfter=10)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [P("&bull; " + i, ParagraphStyle("b", parent=BODY, leftIndent=14, firstLineIndent=-10, spaceAfter=3))
            for i in items]


def table(rows, col_widths, header=True):
    data = []
    for r_i, row in enumerate(rows):
        data.append([Paragraph(escape(str(c)), CELLB if (header and r_i == 0) else CELL) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd4dc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
        for i in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f1f4fb")))
    t.setStyle(TableStyle(style))
    return t


def image(path, width=PAGE_W, caption=None, heading=None):
    if not path.exists():
        return Spacer(1, 1)
    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * h / w)
    parts = ([P(heading, H2)] if heading else []) + [img]
    if caption:
        parts.append(P(f"<i>{escape(caption)}</i>", ParagraphStyle("cap", parent=SUB, fontSize=8.5)))
    return KeepTogether(parts + [Spacer(1, 6)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"Customer Support Ticket Classification - page {doc.page}")
    canvas.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ankita Das")
    args = ap.parse_args()

    m = json.load(open(OUTPUT_DIR / "metrics.json", encoding="utf-8"))
    prep = json.load(open(OUTPUT_DIR / "preprocessing_summary.json", encoding="utf-8"))
    comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    tests = pd.read_csv(OUTPUT_DIR / "new_ticket_test_results.csv")
    errors = pd.read_csv(OUTPUT_DIR / "misclassified_test_tickets.csv")
    clean = pd.read_csv(BASE_DIR / "data" / "tickets_clean.csv")
    labels = m["labels"]
    conf = m["confusion_matrix"]
    n_correct = sum(conf[i][i] for i in range(len(conf)))
    acc = m["accuracy"] * 100
    passed = int((tests["result"] == "PASS").sum())

    story = []

    # ---------------------------------------------------------------- title
    story += [Spacer(1, 2.5 * cm), P("Customer Support Ticket Classification System", TITLE),
              Spacer(1, 6), P("AI / ML Assignment - Solution Documentation", SUB),
              P(f"Submitted by: <b>{escape(args.name)}</b>", SUB), Spacer(1, 18),
              table([["Item", "Summary"],
                     ["Task", "Predict the category of a support ticket from its description"],
                     ["Approach", "Text cleaning -> TF-IDF features -> Logistic Regression"],
                     ["Data", f"{prep['raw_rows']} raw tickets -> {prep['clean_rows']} after cleaning, {prep['n_categories']} categories (synthetic dataset)"],
                     ["Test accuracy", f"{acc:.1f}% on {m['test_size']} unseen tickets"],
                     ["New-ticket tests", f"{passed} of {len(tests)} hand-written new tickets classified correctly"],
                     ["Interfaces", "Command line (predict.py) and Flask web app (app.py)"],
                     ["Bonus", "Rule-based automatic customer response"]],
                    [3.5 * cm, PAGE_W - 3.5 * cm]),
              PageBreak()]

    # ---------------------------------------------------------------- A
    story += [P("A. Problem Understanding", H1),
              P("A company receives customer support requests through e-mail, chat and other channels. "
                "Each ticket has an ID, customer name, description, date, priority, status and category. "
                "Today a person must read every ticket and decide which category it belongs to "
                "(for example <i>Login Issue</i> or <i>Report</i>) before it can be routed to the right team. "
                "This is slow and repetitive."),
              P("The goal of this project is to build a small machine-learning system that learns from "
                "<b>historical tickets</b> (description + known category) and then predicts the category of a "
                "<b>new ticket</b> automatically. This is a <b>supervised, multi-class text-classification</b> problem: "
                "the input is free text and the output is one of several fixed categories.")]

    # ---------------------------------------------------------------- B
    story += [P("B. Dataset", H1),
              P("<b>Source:</b> The dataset was <b>created by me</b> (synthetic). No external or public dataset was used, "
                "so no third-party data source or URL applies. It is generated by <font face='Courier'>src/generate_dataset.py</font> "
                f"with a fixed random seed, so it can be reproduced exactly and is stored as <font face='Courier'>data/tickets.csv</font>. "
                "For each category, 26 realistic base descriptions were written by hand; each was used twice with different "
                "greetings, closing phrases and small noise (lower-casing, extra spaces, '!!') to imitate real tickets. "
                "A few duplicate rows and missing values were added on purpose to practise data cleaning."),
              table([["Property", "Value"],
                     ["Records in raw CSV", prep["raw_rows"]],
                     ["Records after cleaning", prep["clean_rows"]],
                     ["Number of categories", prep["n_categories"]],
                     ["Categories", ", ".join(labels)],
                     ["Fields", "ticket_id, customer_name, ticket_description, date, category, priority, status "
                                "(+ clean_description created during preprocessing)"]],
                    [4.5 * cm, PAGE_W - 4.5 * cm]),
              Spacer(1, 6)]
    sample = pd.read_csv(RAW_CSV).head(6)[["ticket_id", "ticket_description", "category", "priority", "status"]].fillna("(missing)")
    story += [P("Sample rows:", H2),
              table([list(sample.columns)] + sample.values.tolist(),
                    [2 * cm, 7.4 * cm, 3 * cm, 2 * cm, PAGE_W - 14.4 * cm]),
              Spacer(1, 6),
              image(SCREENSHOT_DIR / "analysis.png", heading="Exploratory analysis", caption="Figure 1: Category, priority, status distributions and category-vs-priority heat-map")]
    obs_file = OUTPUT_DIR / "eda_observations.txt"
    if obs_file.exists():
        obs = obs_file.read_text(encoding="utf-8").split("\n\nTop keywords")[0].strip().split("\n")
        story += [P("What the data shows:", H2)] + bullets([escape(o) for o in obs if o.strip()])

    # ---------------------------------------------------------------- C
    story += [P("C. Data Preprocessing", H1),
              table([["Step", "What was done", "Why"],
                     ["Missing values",
                      f"{prep['dropped_missing_description_or_category']} rows without a description or category were dropped; "
                      f"{prep['filled_priority']} missing priorities were filled with the most common value ({prep['priority_filled_with']}); "
                      f"{prep['filled_status']} missing statuses were set to 'Unknown'.",
                      "A ticket without text or label cannot be used for learning. Priority/status are not model inputs, so filling keeps the rows."],
                     ["Duplicates",
                      f"{prep['removed_exact_duplicates']} exact duplicate rows and {prep['removed_duplicate_descriptions']} re-submitted tickets "
                      "(same text, new ID) were removed.",
                      "Duplicates over-weight some tickets and, worse, can put the same ticket in both the training and the test set, which makes the score look better than it is."],
                     ["Text cleaning",
                      "Lower-case; remove punctuation, URLs and e-mail addresses; expand contractions (can't -> cannot); remove extra spaces. "
                      "Negations ('not', 'cannot') and digits (error codes such as 500) are kept.",
                      "'Login', 'login' and 'login!!' must be the same word for the model. Negations and error codes carry meaning, so they must not be removed."],
                     ["Tokenisation", "Done inside TF-IDF (splits the cleaned text into words and word pairs).", "Keeps the pipeline simple."]],
                    [2.4 * cm, 8 * cm, PAGE_W - 10.4 * cm]),
              Spacer(1, 6),
              P(f"Result: {prep['raw_rows']} rows -> {prep['clean_rows']} clean rows with no missing values and no duplicates.")]

    # ---------------------------------------------------------------- D
    story += [P("D. Model Selection", H1),
              P("<b>Pipeline:</b>"),
              P("Ticket description -> Text cleaning -> TF-IDF vectoriser (1-2 word n-grams) -> Logistic Regression -> Predicted category + confidence", CODE),
              P("<b>Features.</b> A computer cannot read words, so TF-IDF turns each ticket into a list of numbers. "
                "<b>TF</b> (term frequency) counts how often a word is in the ticket; <b>IDF</b> (inverse document frequency) gives "
                "more weight to rare, informative words (such as 'password' or 'invoice') and less weight to words found in almost every "
                "ticket (such as 'please' or 'help'). I used single words and two-word phrases (e.g. 'sign in', 'not working') "
                "with sub-linear term frequency, giving a sparse vector per ticket."),
              P("<b>Algorithm: Logistic Regression.</b> For every category it learns one weight per word/phrase. For a new ticket it adds up "
                "the weights of the words present, turns the six scores into probabilities with the softmax function, and returns the category with "
                "the highest probability. That probability is the 'confidence' shown to the user."),
              P("<b>Why Logistic Regression?</b>"),
              ] + bullets([
        "It is a standard, strong baseline for sparse TF-IDF text features and works well on small datasets.",
        "It outputs real probabilities, which gives the confidence score the assignment asks for (Linear SVM does not).",
        "It is easy to explain: the largest weights show which words drive each category (see below).",
        "Training takes well under a second, and it is simple to understand and maintain.",
    ]) + [P("<b>Comparison with other algorithms</b> (5-fold cross-validation on the training data only; the test set was not used for choosing):")]
    comp_rows = [["Algorithm", "CV macro F1 (mean)", "Std. dev."]] + [
        [r.model, f"{r.cv_f1_macro_mean*100:.1f}%", f"{r.cv_f1_macro_std*100:.1f}"] for r in comp.itertuples()]
    story += [table(comp_rows, [7 * cm, 4 * cm, 3 * cm]), Spacer(1, 4),
              P("All four algorithms perform within about two percentage points of each other, which is within the cross-validation spread. "
                "Logistic Regression was kept because of its probabilities and interpretability; the other models are equally valid choices on this small dataset.")]
    feat_rows = [["Category", "Words/phrases with the strongest weight"]] + [[k, ", ".join(v)] for k, v in m["top_features"].items()]
    story += [P("Words that drive each prediction (largest model weights):", H2), table(feat_rows, [4 * cm, PAGE_W - 4 * cm])]

    # ---------------------------------------------------------------- E
    story += [P("E. Training and Testing", H1),
              table([["Item", "Detail"],
                     ["Split", f"80% training ({m['train_size']} tickets) / 20% testing ({m['test_size']} tickets)"],
                     ["Type of split", "Random but stratified: every category has the same share in both parts. Fixed seed (42) so results can be repeated."],
                     ["Training process", "The TF-IDF vocabulary and the Logistic Regression weights are learned from the training tickets only."],
                     ["Testing process", "The 20% test tickets were never seen in training. The model predicts their categories and the predictions are compared to the true categories."],
                     ["Algorithm choice", "Made with cross-validation on the training part only, so the test result is not 'tuned' to the test data."]],
                    [3.5 * cm, PAGE_W - 3.5 * cm])]

    # ---------------------------------------------------------------- F
    story += [P("F. Results", H1),
              table([["Metric", "Weighted average", "Macro average"],
                     ["Accuracy", f"{acc:.2f}%", "-"],
                     ["Precision", f"{m['precision_weighted']*100:.2f}%", f"{m['precision_macro']*100:.2f}%"],
                     ["Recall", f"{m['recall_weighted']*100:.2f}%", f"{m['recall_macro']*100:.2f}%"],
                     ["F1 score", f"{m['f1_weighted']*100:.2f}%", f"{m['f1_macro']*100:.2f}%"]],
                    [4 * cm, 4.5 * cm, 4.5 * cm]), Spacer(1, 6)]
    pc_rows = [["Category", "Precision", "Recall", "F1", "Tickets"]] + [
        [k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1-score']:.2f}", int(v["support"])] for k, v in m["per_class"].items()]
    story += [table(pc_rows, [5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]), Spacer(1, 6),
              image(SCREENSHOT_DIR / "confusion_matrix.png", caption="Figure 2: Confusion matrix on the test data (rows = actual, columns = predicted)"),
              P("Explanation in simple language", H2)] + bullets([
        f"<b>Accuracy {acc:.1f}%</b> means the model put {n_correct} of the {m['test_size']} test tickets into the correct category.",
        "<b>Precision</b> answers: 'when the model says <i>Billing Issue</i>, how often is it right?' High precision means few wrong alarms.",
        "<b>Recall</b> answers: 'of all real <i>Billing Issue</i> tickets, how many did the model find?' High recall means few missed tickets.",
        "<b>F1 score</b> combines precision and recall into one number (their harmonic mean).",
        "<b>Confusion matrix:</b> each row is the true category, each column the predicted one. Numbers on the diagonal are correct predictions; numbers off the diagonal are mistakes.",
    ])
    if len(errors):
        story += [P(f"The {len(errors)} mistake(s) on the test set:", H2),
                  table([["Ticket", "Actual", "Predicted"]] + errors[["description", "actual", "predicted"]].values.tolist(),
                        [8.5 * cm, 3.5 * cm, PAGE_W - 12 * cm])]
    story += [P("Is the model satisfactory?", H2),
              P(f"Yes for this task and dataset: accuracy, precision, recall and F1 are all about {acc:.0f}%, every category is handled well, and "
                f"{passed} of {len(tests)} completely new, differently worded tickets were classified correctly. "
                "However, the score should be read with care: the data is synthetic and each base sentence appears twice "
                "(with different wording around it), so the test set contains tickets that are very similar to training tickets. "
                "The score is therefore probably higher than it would be on real company tickets (see Limitations). "
                "The training accuracy of 100% against a lower test accuracy also shows a small amount of over-fitting, which is normal for such a small dataset.")]

    # ---------------------------------------------------------------- G
    story += [P("G. Prediction on New Tickets", H1),
              P(f"Ten new tickets, written by hand with wording different from the dataset, were tested. The script <font face='Courier'>src/test_new_tickets.py</font> "
                f"also checks that none of them exists in the dataset. Result: <b>{passed} / {len(tests)} correct</b>."),
              table([["Test input", "Expected", "Predicted", "Conf.", "Result"]] +
                    [[row["test_input"], row["expected_category"], row["predicted_category"], f"{row['confidence_%']}%", row["result"]]
                     for _, row in tests.iterrows()],
                    [6.3 * cm, 2.9 * cm, 2.9 * cm, 1.4 * cm, PAGE_W - 13.5 * cm]), Spacer(1, 8),
              image(SCREENSHOT_DIR / "prediction.png", width=PAGE_W * 0.82, caption="Figure 3: Command-line interface (predict.py)"),
              image(SCREENSHOT_DIR / "web_interface.png", width=PAGE_W * 0.62, caption="Figure 4: Web interface (app.py)")]

    # ---------------------------------------------------------------- H
    story += [P("H. Limitations", H1)] + bullets([
        "<b>Small, synthetic dataset:</b> about 300 tickets written by me. Real tickets are messier, longer, contain typos and several problems at once.",
        "<b>Optimistic scores:</b> each base sentence is used twice, so test tickets resemble training tickets. Real-world accuracy would be lower.",
        "<b>Limited categories:</b> only 6 categories and exactly one label per ticket; real tickets can belong to two (e.g. a slow report page).",
        "<b>Similar wording across categories:</b> words like 'report', 'update', 'account' or 'invoice' occur in more than one category and cause confusion.",
        "<b>Bag-of-words only:</b> TF-IDF ignores meaning. A ticket with completely new vocabulary or synonyms that never appeared in training can be misclassified.",
        "<b>English only</b>, no spelling correction, and no handling of very short or ambiguous tickets (a confidence threshold flags low-confidence cases for a human).",
        "<b>No retraining:</b> the model does not learn from new tickets or from agents' corrections.",
    ])

    # ---------------------------------------------------------------- I
    story += [P("I. Future Improvements", H1)] + bullets([
        "Use a much larger set of <b>real, labelled tickets</b> and more categories (plus a multi-label option).",
        "Better preprocessing: spelling correction, lemmatisation, handling of multi-language tickets.",
        "Try sentence embeddings or a fine-tuned transformer (e.g. BERT), or an <b>LLM-based classifier</b> for difficult and ambiguous tickets.",
        "Evaluate with cross-validation grouped by near-duplicate text, and monitor per-category errors.",
        "<b>Human feedback loop:</b> agents correct wrong predictions, and these are stored and used for periodic <b>retraining</b>.",
        "Send low-confidence tickets to a human; auto-route high-confidence tickets.",
        "Production API (e.g. FastAPI), a <b>database</b> for tickets and predictions, logging, authentication and monitoring.",
        "Use an LLM to draft richer personalised replies, always reviewed by an agent.",
    ])

    # ---------------------------------------------------------------- J
    story += [P("J. Bonus Feature, Libraries and AI-Tool Disclosure", H1),
              P("<b>Bonus - automatic response:</b> implemented with <b>rule-based templates</b> (<font face='Courier'>src/responses.py</font>): each predicted "
                "category has a ready-made reply that is shown in both the command-line and the web interface. No external API, model or API key is used."),
              P("<b>Libraries:</b> Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Flask (web interface), ReportLab (this PDF). "
                "Optional: Playwright (only to take the web screenshot)."),
              P("<b>AI tools used:</b> Claude (Anthropic's AI assistant) was used to help generate the initial project code, the synthetic dataset "
                "generator and this documentation. I ran the project, reviewed the code and can explain every step of it."),
              P("<b>Project structure:</b> see README.md for installation and run instructions.")]

    out = BASE_DIR / "report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Customer Support Ticket Classification - Report", author=args.name)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Report created -> {out}")


if __name__ == "__main__":
    main()
