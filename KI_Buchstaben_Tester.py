# predict.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Funktion zum Laden und Vorverarbeiten eines einzelnen Testbildes
def load_and_preprocess_image(image_path, img_size=28):
    # Bild einlesen
    image = cv2.imread(image_path, 0)  # 0 bedeutet Graustufen
    if image is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        return None

    # Bild auf 28x28 skalieren
    resized_img = cv2.resize(image, (img_size, img_size))

    # Normalisieren der Bildwerte auf den Bereich [0, 1]
    normalized_img = resized_img / 255.0

    # Umwandeln in das richtige Format für das Modell (28x28x1)
    img_array = normalized_img.reshape(-1, 28, 28, 1).astype('float32')

    return img_array

# Lade das Modell
model = load_model('mein_model.h5')
print("Modell geladen!")

# Pfad zum Testbild
image_path = r"C:\1BHELHTLInn\4BHEL\KISY_Korber\Tensaflow\Alphabet_recognizer\Buchstaben\einzeln\Alimgeriev_G1.png"


# Bild vorverarbeiten
image_array = load_and_preprocess_image(image_path)

if image_array is not None:
    # Vorhersage für das Bild treffen
    prediction = model.predict(image_array)
    print(prediction)
    # Ermitteln des Indexes der höchsten Wahrscheinlichkeit
    predicted_label = np.argmax(prediction)

    # Umwandeln des Indexes zurück in den Buchstaben
    predicted_char = chr(predicted_label + 65)  # Umwandeln von 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'
    confidence = np.max(prediction) * 100  # Wahrscheinlichkeit in Prozent

    print(f"Vorhergesagter Buchstabe: {predicted_char} ({confidence:.2f}%)")