import os

from flask import Flask, render_template, request

app = Flask(__name__)

ANIMALS = ["Bear", "Elk", "Salmon", "Hawk", "Fox"]
SCORE_CARDS = ["A", "B", "C", "D"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/auto-score", methods=["GET", "POST"])
def auto_score():
    if request.method == "POST":
        cards = {animal: request.form.get(f"{animal.lower()}_card", "A") for animal in ANIMALS}
        file = request.files.get("board_image")
        filename = file.filename if file and file.filename else None
        return render_template(
            "auto_score.html",
            animals=ANIMALS,
            score_cards=SCORE_CARDS,
            submitted=True,
            cards=cards,
            filename=filename,
        )
    return render_template("auto_score.html", animals=ANIMALS, score_cards=SCORE_CARDS)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
