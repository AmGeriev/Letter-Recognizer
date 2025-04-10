# train.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model  # Importiere das Modell aus model.py


# Funktion zum Laden und Vorverarbeiten der Bilder
def load_and_preprocess_images(folder, img_size=28):
    images = []  # Liste für die Bilder
    labels = []  # Liste für die Labels

    # Durchlaufe alle Ordner im Hauptordner (A-Z)
    for letter_folder in os.listdir(folder):
        letter_path = os.path.join(folder, letter_folder)  # Vollständiger Pfad zum Ordner

        # Überprüfen, ob es ein Ordner ist
        if os.path.isdir(letter_path):
            # Durchlaufe alle Bilder im Ordner
            for filename in os.listdir(letter_path):
                filepath = os.path.join(letter_path, filename)  # Erstelle den vollständigen Pfad

                # Überprüfen, ob es eine Datei ist
                if os.path.isfile(filepath):
                    # Bild im Graustufenmodus einlesen
                    image = cv2.imread(filepath, 0)

                    # Bild auf 28x28 skalieren
                    resized_img = cv2.resize(image, (img_size, img_size))

                    # Normalisieren der Bildwerte auf den Bereich [0, 1]
                    normalized_img = resized_img / 255.0

                    # Bild und Label hinzufügen
                    images.append(normalized_img)
                    labels.append(letter_folder)  # Verwende den Ordnernamen (Buchstaben) als Label

    # Umwandeln der Liste in ein NumPy-Array
    images_array = np.array(images)

    # Umwandeln der Labels in numerische Werte (z.B. A=0, B=1, C=2,...)
    # Wir erstellen ein Mapping für die Buchstaben
    label_mapping = {chr(65 + i): i for i in range(26)}  # A-Z -> 0-25
    labels_numeric = np.array([label_mapping[label] for label in labels]).reshape(-1, 1)

    # Rückgabe der Arrays
    return images_array, labels_numeric


# Pfad zum Hauptordner mit den Unterordnern A-Z
folder_path = r"C:\1BHELHTLInn\4BHEL\KISY_Korber\Tensaflow\Alphabet_recognizer\alle_buchstaben"
img_size = 28  # Skalierung auf 28x28 Pixel

# Lade und verarbeite die Bilder und Labels
images_array, labels_array = load_and_preprocess_images(folder_path, img_size)

# Normalisiere die Bilder
images_array = images_array.reshape(-1, 28, 28, 1)  # Anpassung auf die Eingabeform für CNNs
images_array = images_array.astype('float32')  # Um sicherzustellen, dass die Werte als float32 vorliegen
labels_array = labels_array.astype('int32')  # Um sicherzustellen, dass die Labels als Integer vorliegen

# Splitte die Daten in Trainings- und Testdaten
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.1, random_state=42)

# Erstelle das Modell
model = create_model (input_shape=(28, 28, 1), num_classes=26)

# Trainiere das Modell
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Teste das Modell
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Speichern des Modells
model.save("mein_model.h5")
print(f"Model saved")