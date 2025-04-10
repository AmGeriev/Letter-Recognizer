import cv2
import numpy as np
import os
import math

# User parameters
output_folder = "A_bis_O"
image = cv2.imread("buchstaben.jpg")
number_cols = 8  # Anzahl der Spalten (Buchstaben in einer Zeile)
number_rows = 15  # Anzahl der Reihen (Buchstaben in einer Spalte)
margin = 2  # Rand, der weggeschnitten wird bei den einzelnen Buchstaben

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width, channels = image.shape
print(height)
print(width)

# Berechne die genaue Breite und Höhe eines Buchstabens
letter_width = math.ceil(width / number_cols)
letter_height = math.ceil(height / number_rows)

print(letter_width)
print(letter_height)

# Buchstaben, die im Bild verwendet werden (von P bis Z)
alphabet = "ABCDEFGHIJKLMNO"

# Stelle sicher, dass der Ausgabeordner existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Durchlaufe alle Zeilen und Spalten und extrahiere die Buchstaben
letter_counter = 0
for i in range(number_rows):
    for j in range(number_cols):
        x = j * letter_width
        y = i * letter_height

        # Ausschneiden des Buchstabens mit Berücksichtigung des Randes
        letter_img = gray[y + margin:y + letter_height - margin, x + margin:x + letter_width - margin]

        # Sicherstellen, dass das Bild tatsächlich einen Inhalt hat
        if letter_img.size > 0:
            # Optional: Skalierung des Buchstabens auf eine feste Größe
            letter_img_resized = cv2.resize(letter_img, (letter_width - 2 * margin, letter_height - 2 * margin))

            # Bestimmen des aktuellen Buchstabens
            letter = alphabet[letter_counter // number_cols]  # Hole den Buchstaben (P bis Z)

            # Bestimme die Nummer des Bildes für jeden Buchstaben (1 bis 8 für P, 1 bis 8 für Q, ... usw.)
            letter_number = (letter_counter % number_cols) + 1

            # Erstelle für jeden Buchstaben einen eigenen Unterordner, falls er nicht existiert
            letter_folder = os.path.join(output_folder, letter)
            if not os.path.exists(letter_folder):
                os.makedirs(letter_folder)

            # Benennen der Dateien wie Alimgeriev_P1, Alimgeriev_P2, ..., Alimgeriev_Q1, ..., Alimgeriev_Z8
            letter_filename = f"Alimgeriev_{letter}{letter_number}.png"

            # Speichern des Buchstabenbildes im entsprechenden Ordner
            cv2.imwrite(os.path.join(letter_folder, letter_filename), letter_img_resized)

            letter_counter += 1

# cv2.imshow('BuchstabenRaster', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
