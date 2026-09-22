import io

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def preprocess_for_ocr(image_rgb):
    """Ameliore la lisibilite avant OCR : niveaux de gris, agrandissement,
    debruitage leger et binarisation adaptative (utile pour l'ecriture
    manuscrite photographiee, souvent avec un eclairage inegal)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Agrandir les petites photos pour aider Tesseract sur les traits fins
    height, width = gray.shape
    if max(height, width) < 1800:
        scale = 1800 / max(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return binary


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def extract_text():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image recue."}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Fichier image invalide."}), 400

    image_np = np.array(image)
    processed = preprocess_for_ocr(image_np)

    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(processed, lang="fra+eng", config=config)
    except pytesseract.TesseractNotFoundError:
        return jsonify({"error": "Tesseract OCR n'est pas installe sur le serveur."}), 500

    text = text.strip()
    if not text:
        return jsonify({
            "text": "",
            "warning": "Aucun texte detecte. Essaie une photo plus nette, mieux eclairee et bien cadree sur l'ecriture.",
        })

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
