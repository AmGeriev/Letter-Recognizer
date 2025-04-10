# Alphabet Recognizer - README

## Übersicht
Dieses Projekt dient zur Erkennung von handschriftlichen Buchstaben mithilfe eines KI-Modells. Es besteht aus mehreren Python-Skripten, die aufeinander aufbauen, um Bilder zu verarbeiten, ein Modell zu trainieren und dieses zu testen.

## Benötigte Libraries
Bitte stelle sicher, dass du folgende Libraries installiert hast:

```bash
pip install numpy
pip install opencv-python
pip install tensorflow
pip install matplotlib
pip install pillow
```

## Ablauf

### 1. Buchstaben extrahieren
Führe zuerst das Skript `Buchstaben_extrahieren.py` aus. Dieses schneidet die Bilder in einzelne Buchstabenformate und speichert sie ab.

### 2. Bilder in NumPy umwandeln
Nutze das Skript `Numpy_umwandler.py`, um die gespeicherten Buchstabenbilder in NumPy-Arrays umzuwandeln.

### 3. Modell trainieren
Führe das Skript `train_model.py` aus, um dein KI-Modell zu trainieren. Das trainierte Modell wird automatisch gespeichert.

### 4. Modell testen (optional)
Teste dein Modell mit einzelnen Buchstaben durch das Skript `KI_Buchstaben_Tester.py`.

### 5. Grafische Benutzeroberfläche
Starte das Skript `tkinter_model.py`, um das Modell über eine grafische Oberfläche mit TKinter zu nutzen und zu testen.

## Hinweise
- Achte darauf, die Dateipfade in den Skripten ggf. anzupassen.
- Alle Trainingsdaten und Modellinformationen sollten im gleichen Projektordner liegen.

Viel Erfolg beim Testen und Trainieren deines Alphabet-Erkenners!
