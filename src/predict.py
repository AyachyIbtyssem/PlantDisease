import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# === 1. Charger le modèle entraîné ===
model_path = 'models/vgg16_finetuned.h5'  # chemin vers ton modèle
model = tf.keras.models.load_model(model_path)
print("✅ Modèle chargé avec succès !")

# === 2. Définir le chemin vers le dossier train pour récupérer les classes ===
train_dir = 'data/Chili_Plant_Disease/train'  # adapte selon ton dataset
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"📚 Classes détectées : {class_names}")

# === 3. Fonction de prédiction ===
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Le fichier {img_path} n'existe pas.")
        return

    # Charger et prétraiter l'image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prédiction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]

    # Affichage
    print(f"🔍 Image : {os.path.basename(img_path)}")
    print(f"🌿 Classe prédite : {predicted_class}")

# === 4. Exemple ===
example_image = 'data/Chili_Plant_Disease/test/healthy/Cabai sehat003.jpg'  
predict_image(example_image)
