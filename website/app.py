import os

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

ANIMALS = ["Bear", "Elk", "Salmon", "Hawk", "Fox"]
SCORE_CARDS = ["A", "B", "C", "D"]

ANIMAL_SCORES = {
    "Bear": 12,
    "Elk": 9,
    "Salmon": 15,
    "Hawk": 7,
    "Fox": 11,
}

HABITATS = ["Mountains", "Forests", "Prairies", "Wetlands", "Rivers"]
HABITAT_SCORES = {
    "Mountains": 10,
    "Forests": 14,
    "Prairies": 8,
    "Wetlands": 11,
    "Rivers": 9,
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/auto-score", methods=["GET", "POST"])
def auto_score():
    if request.method == "POST":
        return redirect(url_for("result"))
    return render_template("auto_score.html", animals=ANIMALS, score_cards=SCORE_CARDS)


@app.route("/result")
def result():
    animal_total = sum(ANIMAL_SCORES.values())
    habitat_total = sum(HABITAT_SCORES.values())
    total = animal_total + habitat_total
    solo_bonus = sum(2 for score in HABITAT_SCORES.values() if score > 7)
    return render_template(
        "result.html",
        animals=ANIMALS,
        animal_scores=ANIMAL_SCORES,
        animal_total=animal_total,
        habitats=HABITATS,
        habitat_scores=HABITAT_SCORES,
        habitat_total=habitat_total,
        solo_bonus=solo_bonus,
        total=total,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
