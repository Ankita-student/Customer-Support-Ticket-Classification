# Customer Support Ticket Classification System

An AI/ML project that learns from historical support tickets and automatically predicts the
**category** of a new ticket from its description.

```
Ticket Description -> Text Cleaning -> TF-IDF -> Logistic Regression -> Predicted Category (+ confidence)
```

## 1. Project Overview

| Item | Details |
|---|---|
| Task | Multi-class text classification (6 categories) |
| Model | TF-IDF (1-2 word n-grams) + Logistic Regression (scikit-learn) |
| Data | 322 raw synthetic tickets -> 308 after cleaning |
| Test accuracy | ~96.8% on 62 unseen tickets (see `report.pdf` for precision / recall / F1 / confusion matrix) |
| Interfaces | Command line (`predict.py`) and Flask web app (`app.py`) |
| Bonus | Rule-based automatic customer response (no API, no keys) |

## 2. Problem Statement

A company receives support requests by e-mail, chat, etc. Each ticket must be assigned to a category
(*Login Issue, Application Error, Report, Account Update, Performance, Billing Issue*) so it reaches the right team.
Doing this by hand is slow. The system learns from past tickets and predicts the category of a new ticket automatically.

## 3. Technologies Used

Python 3.9+, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Flask, ReportLab (for `report.pdf`).
Optional: Playwright (only to auto-capture the web-page screenshot).

## 4. Dataset Information

* **Source:** created by me (synthetic) with `src/generate_dataset.py`. **No external dataset** is used, so there is no third-party source URL.
* 26 hand-written base descriptions per category, each used twice with different greetings/closings/noise -> 312 tickets, plus 10 deliberately added duplicates and 11 missing values to practise cleaning (322 raw rows).
* **Fields:** `ticket_id, customer_name, ticket_description, date, category, priority, status`
* **Categories (6):** Login Issue, Application Error, Report, Account Update, Performance, Billing Issue
* `data/tickets.csv` = raw data, `data/tickets_clean.csv` = cleaned data (created by `preprocessing.py`).

## 5. Installation Steps

You need **Python 3.9 or newer**. Open a terminal **inside this project folder**.

```bash
# (recommended) create a virtual environment
python -m venv venv

# activate it:
#   Windows (cmd / PowerShell):   venv\Scripts\activate
#   macOS / Linux:                source venv/bin/activate

pip install -r requirements.txt
```

> On macOS/Linux you may need to type `python3` / `pip3` instead of `python` / `pip`.

## 6. Required Dependencies

Listed in `requirements.txt`: `pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, flask, reportlab`.

## 7. How to Train the Model

The project already contains the data, a trained model, the charts and the report. To (re)create everything yourself:

```bash
# Everything in one command (data -> cleaning -> EDA -> training -> tests -> screenshots -> report.pdf)
python src/run_pipeline.py --name "Ankita Das"

# ...or step by step:
python src/generate_dataset.py     # creates data/tickets.csv
python src/preprocessing.py        # Task 1: cleaning       -> data/tickets_clean.csv
python src/eda.py                  # Task 2: analysis+charts -> screenshots/
python src/train.py                # Task 3+4: train + evaluate -> model/ + outputs/
python src/test_new_tickets.py     # Task 5: 10 new-ticket tests
python src/make_screenshots.py     # screenshots for the report
python src/make_report.py --name "Ankita Das"   # Task 7: report.pdf
```

Training prints accuracy, precision, recall, F1, the confusion matrix and the mistakes, and saves the model to `model/ticket_classifier.joblib`.
To use your **own** CSV, put it at `data/tickets.csv` (same columns) and run `python src/run_pipeline.py --skip-generate`.

## 8. How to Perform Prediction

```bash
# interactive (type tickets one by one, 'q' to quit)
python src/predict.py

# single prediction
python src/predict.py --text "I forgot my password and cannot login"
```

If the saved model is missing or was created with a different scikit-learn version, it is **re-trained automatically** (a few seconds).

## 9. How to Run the Application (Web Interface)

```bash
python src/app.py
```

Open **http://127.0.0.1:5000** in your browser, type a ticket, click **Predict**. Stop the server with `Ctrl + C`.
(To use another port: `PORT=8080 python src/app.py`; on Windows PowerShell: `$env:PORT=8080; python src/app.py`.)

A JSON API is also available:

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
     -d '{"description": "I was charged twice for my subscription"}'
```

## 10. Sample Input / Output

```
====================================
 Customer Support Classifier
====================================
Enter Ticket Description:
> I am unable to login because my password is not working.

Predicted Category:
  Login Issue
Confidence: 96.1%

Suggested Response:
  Please use the 'Forgot Password' option on the login page to reset your password. ...
```

Tests on 10 new tickets (excerpt; full table in `report.pdf` and `outputs/new_ticket_test_results.csv`):

| Test input | Expected | Predicted | Confidence |
|---|---|---|---|
| Everything is crawling, pages are taking ages to open | Performance | Performance | 82.7% |
| I was debited twice for the same invoice | Billing Issue | Billing Issue | 75.7% |
| Please send me the sales figures summary for October | Report | Report | 69.0% |

## Project Structure

```
AI_Assignment_Ankita Das/
├── data/
│   ├── tickets.csv              raw dataset
│   └── tickets_clean.csv        cleaned dataset
├── src/
│   ├── config.py                paths + constants
│   ├── generate_dataset.py      builds the synthetic dataset
│   ├── preprocessing.py         Task 1 - cleaning (+ clean_text used everywhere)
│   ├── eda.py                   Task 2 - analysis and charts
│   ├── train.py                 Task 3+4 - model training and evaluation
│   ├── predict.py               Task 5 + CLI interface (Task 6A)
│   ├── test_new_tickets.py      Task 5 - documented tests on new tickets
│   ├── app.py + templates/      Task 6B - Flask web interface
│   ├── responses.py             Bonus - rule-based customer reply
│   ├── make_screenshots.py      creates the screenshots
│   ├── make_report.py           Task 7 - creates report.pdf
│   └── run_pipeline.py          runs everything
├── model/                       trained model (ticket_classifier.joblib)
├── outputs/                     metrics.json, classification report, test results ...
├── screenshots/                 dataset, analysis, confusion matrix, prediction, web UI
├── requirements.txt
├── README.md
└── report.pdf                   solution documentation (Task 7)
```

## Mapping to the Assignment Tasks

| Task | Where |
|---|---|
| 1 Data preparation | `src/preprocessing.py` |
| 2 EDA (5 charts) | `src/eda.py` -> `screenshots/` |
| 3 Model | `src/train.py` (explained in `report.pdf`, section D) |
| 4 Training / testing | `src/train.py` -> `outputs/metrics.json`, `screenshots/confusion_matrix.png` |
| 5 New ticket prediction | `src/predict.py`, `src/test_new_tickets.py` |
| 6 Interface | CLI: `src/predict.py`  -  Web: `src/app.py` |
| 7 Documentation | `report.pdf` |
| Bonus | `src/responses.py` (rule-based, no API) |

## Limitations (short)

Small synthetic dataset; each base sentence appears twice so the test score is optimistic; only 6 single-label categories;
bag-of-words ignores meaning; English only. Details and future improvements are in `report.pdf` (sections H and I).

## AI Tools Disclosure

Claude (Anthropic's AI assistant) was used to help generate the initial project code, the synthetic dataset generator and the
documentation. The code was run and checked, and the author should be able to explain it (see the notes below).
**No API keys, passwords or tokens are included in this project.**

## Notes for the Technical Discussion (be ready to explain)

* **Why clean the text?** So "Login", "login" and "login!!" are the same word. Negations (`not`, `cannot`) and digits are kept on purpose.
* **Why remove duplicates before splitting?** Otherwise the same ticket could be in train *and* test and inflate the score.
* **What is TF-IDF?** Each ticket becomes a vector; rare informative words get higher weight than common words like "please".
* **How does Logistic Regression predict?** It learns one weight per word per category, sums the weights of the words in the ticket, applies softmax to get probabilities and returns the category with the highest one.
* **Precision / recall / F1?** Precision = correct among predicted; recall = found among actual; F1 = their harmonic mean.
* **Why is the score probably optimistic?** Synthetic data, each base sentence used twice -> test tickets are similar to training tickets.
* **What would you improve?** Real data, more categories, embeddings / transformers / LLM, human feedback + retraining, an API and a database.

## Using this project in VS Code

1. **File > Open Folder...** and select the folder `AI_Assignment_Ankita Das`.
2. Open the terminal (**Terminal > New Terminal**) and create/activate a virtual environment, then install packages:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Select the interpreter: `Ctrl+Shift+P` > **Python: Select Interpreter** > choose the `venv` one.
4. Open **Run and Debug** (`Ctrl+Shift+D`) and pick a ready-made configuration from the dropdown, then press the green play button:
   * **1. Run everything (pipeline)** - data, cleaning, EDA, training, tests, screenshots, report.pdf
   * **2. Train model**
   * **3. Predict (command line)** - type tickets in the terminal
   * **4. Web app (Flask)** - then open http://127.0.0.1:5000
5. Or just type the commands from sections 7-9 in the terminal. If your folder path contains a space, use quotes:
   `cd "AI_Assignment_Ankita Das"`.
