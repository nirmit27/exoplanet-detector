import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from keras.models import load_model # type: ignore

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

load_dotenv()

# Loading environment variables
SECRET_KEY = os.environ.get("SECRET_KEY")
MODEL_PATH = os.environ.get("MODEL_PATH")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")

# TODO: Remove the following 3 lines from production
assert SECRET_KEY is not None, "Missing secret key"
assert MODEL_PATH is not None, "Missing model filepath"
assert UPLOAD_FOLDER is not None, "Missing upload folder path"

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
loaded_model = load_model(MODEL_PATH)


def generate_plots(data):
    """Generates Plotly plots for each entry in the dataset."""
    data = pd.read_csv(
        r"uploads\sample_flux_data.csv",
        usecols=lambda x: x != "Unnamed: 0",
    )

    fig = make_subplots(
        rows=data.shape[0],
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"Entry {i + 1}" for i in range(data.shape[0])],
    )

    for index, row in data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[i for i in range(data.shape[1])],
                y=row.values,
                mode="lines",
                name=f"Entry {index + 1}",  # type: ignore
            ),
            row=index + 1,  # type: ignore
            col=1,
        )

    for i in range(1, data.shape[0] + 1):
        fig.update_xaxes(title_text="Time Steps", row=i, col=1)
        fig.update_yaxes(title_text="Flux Intensity", row=i, col=1)

    fig.update_layout(
        height=300 * data.shape[0],
        title="Time Series of Flux Data for Each Entry",
        template="plotly",
        showlegend=False,
    )

    return pio.to_html(fig, full_html=False)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected!", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "error")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename or "")
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            flash("File uploaded successfully!", "success")

            action = request.form.get("action")
            if action == "analyze":
                return redirect(url_for("analyze", filename=filename))
            elif action == "predict":
                return redirect(url_for("predict", filename=filename))

    return render_template("upload.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    filename = request.args.get("filename", None)
    if not filename:
        flash("No file provided for visualization!", "error")
        return redirect(url_for("upload"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        data = pd.read_csv(filepath, header=None)
        plot_html = generate_plots(data)

        return render_template("analyze.html", plot_html=plot_html)

    except Exception as e:
        flash(f"Error processing the file: {str(e)}", "error")
        return redirect(url_for("upload"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    filename = request.args.get("filename", None)
    if not filename:
        flash("No file provided for prediction!", "error")
        return redirect(url_for("upload"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        data = pd.read_csv(filepath, usecols=lambda x: x != "Unnamed: 0")
        if data.shape[1] != 3197:
            flash("The uploaded file must have exactly 3197 features.", "error")
            return redirect(url_for("upload"))

        input_data = data.values
        predictions = (loaded_model.predict(input_data) > 0.5).astype("int32")
        results = [
            {
                "Index": i,
                "Prediction": "Exoplanet Star" if pred == 1 else "Non-exoplanet Star",
            }
            for i, pred in enumerate(predictions)
        ]
        return render_template("results.html", results=results)

    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "error")
        return redirect(url_for("upload"))


@app.route("/results")
def results():
    # TODO Implement user session management

    results = request.args.get("results", [])
    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
