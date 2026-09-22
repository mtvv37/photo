import io
import os
import shutil
import sys

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def locate_tesseract_on_windows():
    """Sur Windows, l'installeur ne modifie pas toujours le PATH : on
    cherche l'executable aux emplacements par defaut pour eviter a
    l'utilisateur de configurer quoi que ce soit."""
    if sys.platform != "win32" or shutil.which("tesseract"):
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


locate_tesseract_on_windows()


OCR_CONFIG = "--oem 3 --psm 6"


def _text_score(gray_image):
    """Estime la qualite d'une orientation/variante en sommant la longueur
    des mots reconnus avec une confiance correcte."""
    data = pytesseract.image_to_data(
        gray_image, lang="fra+eng", config=OCR_CONFIG, output_type=pytesseract.Output.DICT
    )
    score = 0
    for text, conf in zip(data["text"], data["conf"]):
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            continue
        if text.strip() and confidence > 40:
            score += len(text.strip())
    return score


def _crop_to_paper(gray_image, bgr_image):
    """Retire le fond (table, bureau...) autour de la feuille en detectant
    la zone peu saturee en couleur (le papier), pour eviter que la texture
    du fond ne soit prise pour du texte."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    paper_mask = (saturation < 45).astype(np.uint8) * 255
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_image

    largest = max(contours, key=cv2.contourArea)
    height, width = gray_image.shape
    # Si la plus grande zone peu saturee ne couvre presque rien ou presque
    # toute l'image, le fond n'a pas ete detecte correctement : on garde
    # l'image entiere plutot que de mal recadrer.
    area_ratio = cv2.contourArea(largest) / (height * width)
    if area_ratio < 0.15:
        return gray_image

    x, y, w, h = cv2.boundingRect(largest)
    margin = 5
    x0, y0 = max(0, x + margin), max(0, y + margin)
    x1, y1 = min(width, x + w - margin), min(height, y + h - margin)
    if x1 - x0 < 50 or y1 - y0 < 50:
        return gray_image
    return gray_image[y0:y1, x0:x1]


def preprocess_for_ocr(image_rgb):
    """Corrige l'orientation de la photo, recadre sur la feuille et
    binarise le resultat pour maximiser la lisibilite par l'OCR."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Teste les 4 orientations possibles et garde celle qui donne le plus
    # de texte reconnu avec confiance (les photos de notes manuscrites
    # sont tres souvent prises de travers).
    rotations = {
        0: (gray, bgr),
        90: (cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)),
        180: (cv2.rotate(gray, cv2.ROTATE_180), cv2.rotate(bgr, cv2.ROTATE_180)),
        270: (
            cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ),
    }

    best_score, best_gray, best_bgr = -1, gray, bgr
    for rotated_gray, rotated_bgr in rotations.values():
        _, quick_otsu = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        score = _text_score(quick_otsu)
        if score > best_score:
            best_score, best_gray, best_bgr = score, rotated_gray, rotated_bgr

    cropped = _crop_to_paper(best_gray, best_bgr)

    height, width = cropped.shape
    if max(height, width) < 1800:
        scale = 1800 / max(height, width)
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    try:
        text = pytesseract.image_to_string(processed, lang="fra+eng", config=OCR_CONFIG)
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
