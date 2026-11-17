"""Task 2 - Exploratory Data Analysis.

Prints the key counts, produces charts (saved in screenshots/) and prints
automatically generated observations.

Run:  python src/eda.py
"""
import matplotlib

matplotlib.use("Agg")          # save charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from config import CLEAN_CSV, OUTPUT_DIR, SCREENSHOT_DIR, ensure_dirs

sns.set_theme(style="whitegrid")
PRIORITY_ORDER = ["High", "Medium", "Low"]


def load_clean():
    if not CLEAN_CSV.exists():
        import preprocessing
        preprocessing.main()
    return pd.read_csv(CLEAN_CSV)


# ---- individual chart functions (draw on a given axis) ----------------------
def plot_category(df, ax):
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, hue="category", legend=False,
                  palette="viridis", ax=ax)
    ax.set_title("Chart 1: Category Distribution")
    ax.margins(x=0.1)
    ax.set_xlabel("Number of tickets")
    ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, padding=3)


def plot_priority(df, ax):
    sns.countplot(data=df, x="priority", order=PRIORITY_ORDER, hue="priority", legend=False,
                  palette={"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}, ax=ax)
    ax.set_title("Chart 2: Priority Distribution")
    ax.margins(y=0.12)
    ax.set_xlabel("Priority")
    ax.set_ylabel("Number of tickets")
    for c in ax.containers:
        ax.bar_label(c, padding=3)


def plot_status(df, ax):
    order = df["status"].value_counts().index
    sns.countplot(data=df, x="status", order=order, hue="status", legend=False,
                  palette="Set2", ax=ax)
    ax.set_title("Chart 3: Status Distribution")
    ax.margins(y=0.12)
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of tickets")
    for c in ax.containers:
        ax.bar_label(c, padding=3)


def plot_cat_priority(df, ax):
    table = pd.crosstab(df["category"], df["priority"]).reindex(columns=PRIORITY_ORDER, fill_value=0)
    sns.heatmap(table, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_title("Chart 4: Category vs Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("")


def plot_length(df, ax):
    lens = df["clean_description"].str.split().str.len()
    sns.histplot(lens, bins=12, color="#4c78a8", ax=ax)
    ax.set_title("Chart 5: Words per Description")
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Tickets")


def save_single(fn, df, name, size=(7, 4.5)):
    fig, ax = plt.subplots(figsize=size)
    fn(df, ax)
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / name, dpi=130)
    plt.close(fig)


def top_words(df, n=6):
    cv = CountVectorizer(stop_words="english")
    result = {}
    for cat, grp in df.groupby("category"):
        counts = cv.fit_transform(grp["clean_description"]).sum(axis=0).A1
        vocab = cv.get_feature_names_out()
        top = sorted(zip(vocab, counts), key=lambda x: -x[1])[:n]
        result[cat] = [w for w, _ in top]
    return result


def main():
    ensure_dirs()
    df = load_clean()
    print("=" * 70)
    print("TASK 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    total = len(df)
    print(f"\nTotal number of tickets: {total}")
    print("\nTickets per category:")
    print(df["category"].value_counts().to_string())
    print("\nTickets by priority:")
    print(df["priority"].value_counts().reindex(PRIORITY_ORDER).to_string())
    print("\nTickets by status:")
    print(df["status"].value_counts().to_string())
    print("\nCategory distribution (%):")
    print((df["category"].value_counts(normalize=True) * 100).round(1).to_string())

    xt = pd.crosstab(df["category"], df["priority"]).reindex(columns=PRIORITY_ORDER, fill_value=0)
    xt.to_csv(OUTPUT_DIR / "category_vs_priority.csv")
    print("\nCategory vs priority:")
    print(xt.to_string())

    # ---- charts -------------------------------------------------------------
    save_single(plot_category, df, "category_distribution.png")
    save_single(plot_priority, df, "priority_distribution.png")
    save_single(plot_status, df, "status_distribution.png")
    save_single(plot_cat_priority, df, "category_vs_priority.png")
    save_single(plot_length, df, "text_length.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_category(df, axes[0, 0])
    plot_priority(df, axes[0, 1])
    plot_status(df, axes[1, 0])
    plot_cat_priority(df, axes[1, 1])
    fig.suptitle(f"Support Ticket Dataset - Exploratory Analysis ({total} tickets)", fontsize=15)
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "analysis.png", dpi=110)
    plt.close(fig)
    print(f"\nCharts saved in {SCREENSHOT_DIR}")

    # ---- automatic observations --------------------------------------------
    counts = df["category"].value_counts()
    ratio = counts.max() / counts.min()
    high_share = (xt["High"] / xt.sum(axis=1)).sort_values(ascending=False)
    closed_share = (df["status"] == "Closed").mean() * 100
    words = top_words(df)
    lens = df["clean_description"].str.split().str.len()

    obs = [
        f"The dataset is well balanced: the largest category ({counts.index[0]}, {counts.iloc[0]} tickets) "
        f"is only {ratio:.2f}x the size of the smallest ({counts.index[-1]}, {counts.iloc[-1]}), so plain "
        "accuracy is a fair metric and no re-sampling is needed.",
        f"'{high_share.index[0]}' tickets are most often High priority ({high_share.iloc[0]*100:.0f}% of them); "
        f"'{high_share.index[-1]}' tickets are least urgent ({high_share.iloc[-1]*100:.0f}% High).",
        f"Overall {closed_share:.0f}% of tickets are Closed; the rest are still Open or In Progress.",
        f"Descriptions are short (on average {lens.mean():.1f} words), so every keyword matters for the model.",
        "Each category has its own keywords (see below), which is why a bag-of-words / TF-IDF "
        "model is expected to work well.",
        "Some categories share vocabulary (e.g. 'account' appears in both Login Issue and Account Update "
        "tickets, 'update' in Account Update and Billing Issue, and one Performance ticket mentions a "
        "'report' page), so a few confusions are expected.",
        "Filler words such as 'hi', 'help', 'support' and 'thanks' appear in every category; TF-IDF "
        "down-weights words that appear everywhere, so they should not mislead the model.",
    ]
    print("\nIMPORTANT OBSERVATIONS")
    for i, o in enumerate(obs, 1):
        print(f" {i}. {o}")
    print("\nMost frequent keywords per category:")
    for cat, w in words.items():
        print(f"  {cat:<18}: {', '.join(w)}")

    with open(OUTPUT_DIR / "eda_observations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(obs))
        f.write("\n\nTop keywords per category:\n")
        for cat, w in words.items():
            f.write(f"{cat}: {', '.join(w)}\n")


if __name__ == "__main__":
    main()
