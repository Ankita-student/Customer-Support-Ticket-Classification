"""Create the screenshots required for the report.

 - screenshots/dataset.png      first rows of the dataset
 - screenshots/prediction.png   command-line session with new tickets
 - screenshots/web_interface.png  the Flask page (only if 'playwright' + Chromium
                                   are installed - optional, skipped otherwise)

(analysis.png and confusion_matrix.png are created by eda.py and train.py.)

Run:  python src/make_screenshots.py
"""
import textwrap
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RAW_CSV, SCREENSHOT_DIR, ensure_dirs
from predict import format_result, predict_ticket

DEMO_TICKETS = [
    "I am unable to login because my password is not working.",
    "The application is taking too long to load.",
    "Please help me generate my monthly sales report",
    "I was charged twice for my subscription this month",
]


def dataset_image():
    df = pd.read_csv(RAW_CSV).head(14)
    df = df[["ticket_id", "ticket_description", "category", "priority", "status"]].copy()
    df["ticket_description"] = df["ticket_description"].fillna("<missing>").str.slice(0, 62)
    df = df.fillna("<missing>")
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left",
                   colWidths=[0.07, 0.48, 0.14, 0.09, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f5bea")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f4fb")
    total = len(pd.read_csv(RAW_CSV))
    ax.set_title(f"data/tickets.csv - first 14 of {total} rows", fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "dataset.png", dpi=120)
    plt.close(fig)


def prediction_image():
    lines = ["=" * 36, " Customer Support Classifier", "=" * 36,
             "Type a ticket description and press Enter. Type 'q' to quit.", ""]
    for text in DEMO_TICKETS:
        lines.append("Enter Ticket Description:")
        lines.append("> " + text)
        lines.append("")
        for ln in format_result(predict_ticket(text)).splitlines():
            lines.extend(textwrap.wrap(ln, 86, subsequent_indent="  ") or [""])
        lines.append("")
    lines += ["Enter Ticket Description:", "> q", "Bye!"]

    fig_h = 0.19 * len(lines) + 0.6
    fig = plt.figure(figsize=(11, fig_h), facecolor="#1e1e2e")
    fig.text(0.02, 0.985, "\n".join(lines), family="monospace", fontsize=9.5,
             color="#d9e0ee", va="top", ha="left", linespacing=1.35)
    fig.savefig(SCREENSHOT_DIR / "prediction.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def web_image():
    try:
        from playwright.sync_api import sync_playwright
        from werkzeug.serving import make_server
        from app import app
    except ImportError:
        print("  (skipped web_interface.png - 'playwright' is not installed; this is optional)")
        return
    server = make_server("127.0.0.1", 5055, app)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 820, "height": 900})
            page.goto("http://127.0.0.1:5055/")
            page.fill("#description", "The application is very slow today.")
            page.click("button[type=submit]")
            page.wait_for_selector(".result")
            page.screenshot(path=str(SCREENSHOT_DIR / "web_interface.png"), full_page=True)
            browser.close()
    except Exception as e:                      # browser not installed etc.
        print(f"  (skipped web_interface.png - {type(e).__name__}: {str(e)[:80]})")
    finally:
        server.shutdown()


def main():
    ensure_dirs()
    dataset_image()
    print("  dataset.png created")
    prediction_image()
    print("  prediction.png created")
    web_image()
    print(f"Screenshots saved in {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
