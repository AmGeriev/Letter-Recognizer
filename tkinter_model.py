import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw

# Lade das trainierte Modell
model = load_model(r"C:\1BHELHTLInn\4BHEL\KISY_Korber\Tensaflow\Alphabet_recognizer\mein_model.h5")

# Setup der Tkinter GUI
class Zeichnen:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x600")  # Setzt die Fenstergröße auf X * X Pixel
        self.root.title("Buchstaben zeichnen")


        # Canvas erstellen, auf dem man zeichnen kann
        self.canvas = tk.Canvas(root, width= 600, height =600, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)  # Mausklick oder Bewegung für Zeichnen

        # Speichern der Zeichnung
        self.image = Image.new("L", (600, 600), 255)  # "L" steht für Graustufenbild
        self.draw = ImageDraw.Draw(self.image)

        # Button zum Auswerten
        self.auswerten = tk.Button(root, text="Auswerten", font = ("Arial", 24), command=self.predict)
        self.auswerten.pack()

        # Vorhersage-Anzeige
        self.result_label = tk.Label(root, text="Vorhersage: ", font=("Arial", 16))
        self.result_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=30)
        self.draw.line([x1, y1, x2, y2], fill=0, width=10)  # Zeichnen im Bild

    def predict(self):
        # Bild vorverarbeiten
        img_resized = self.image.resize((28, 28))  # Modell erwartet 28x28 Bilder
        img_array = np.array(img_resized) / 255.0  # Normierung
        img_array = np.reshape(img_array, (1, 28, 28, 1))  # Umformen, wie das Modell es erwartet

        # Vorhersage mit dem Modell
        prediction = model.predict(img_array)
        print(prediction)
        predicted_label = np.argmax(prediction)
        accuracy = np.max(prediction) * 100 # Wahrscheinlichkeit in Prozent

        # Ergebnis anzeigen
        self.result_label.config(text=f"Vorhersage: {chr(predicted_label + ord('A'))} ({accuracy:.2f}%)")

        # Zeichnung zurücksetzen
        self.canvas.delete("all")
        self.image = Image.new("L", (600, 600), 255)
        self.draw = ImageDraw.Draw(self.image)








# Hauptfenster
root = tk.Tk()
app = Zeichnen(root)
root.mainloop()
