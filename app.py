"""Task 6 (Option B) - Simple web interface built with Flask.

Run:   python src/app.py
Open:  http://127.0.0.1:5000

JSON API (optional):
    POST /api/predict   body: {"description": "I forgot my password"}
"""
import os

from flask import Flask, jsonify, render_template, request

from predict import load_model, predict_ticket

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    description, result, error = "", None, None
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        if not description:
            error = "Please enter a ticket description."
        else:
            try:
                result = predict_ticket(description)
            except ValueError as e:
                error = str(e)
    return render_template("index.html", description=description, result=result, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return jsonify({"error": "Field 'description' is required."}), 400
    try:
        r = predict_ticket(description)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "category": r["category"],
        "confidence": round(r["confidence"], 4),
        "needs_review": r["needs_review"],
        "suggested_response": r["suggested_response"],
        "top_predictions": [{"category": c, "probability": round(p, 4)} for c, p in r["top"]],
    })


if __name__ == "__main__":
    load_model()                                   # train automatically if needed
    port = int(os.environ.get("PORT", 5000))
    print(f"\nOpen http://127.0.0.1:{port} in your browser (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False)
