# Transcription d'ecriture manuscrite

Mini outil web : upload d'une photo contenant du texte ecrit a la main,
transcription en texte numerique copiable (OCR via Tesseract).

## Installation

Necessite Tesseract OCR (binaire systeme) en plus des dependances Python.

```bash
apt-get install -y tesseract-ocr tesseract-ocr-fra
pip install -r requirements.txt
```

## Lancer l'outil

```bash
python3 app.py
```

Ouvrir http://localhost:5000, charger une photo, cliquer sur
"Transcrire le texte". Le resultat s'affiche dans une zone de texte
avec un bouton "Copier".

## Limites

La reconnaissance d'ecriture manuscrite reste imparfaite, surtout sur
une ecriture cursive ou peu lisible : les meilleurs resultats
s'obtiennent avec une photo nette, bien eclairee, cadree sur le texte,
et une ecriture plutot lisible (lettres detachees). Relire et corriger
le texte transcrit reste recommande.
