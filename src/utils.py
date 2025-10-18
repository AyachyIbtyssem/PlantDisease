import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ========================================
# 🔹 Charger un modèle entraîné
# ========================================
def load_trained_model(model_path='models/vgg16_finetuned.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    model = load_model(model_path)
    print(f"✅ Modèle chargé : {model_path}")
    return model

# ========================================
# 🔹 Récupérer les classes à partir du dossier train
# ========================================
def get_class_names(train_dir):
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dossier introuvable : {train_dir}")
    # Ne garder que les sous-dossiers
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"📚 Classes détectées ({len(class_names)}) : {class_names}")
    return class_names

# ========================================
# 🔹 Prédiction sur une seule image
# ========================================
def predict_single_image(model, img_path, img_size=(224, 224), class_names=None):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    
    img = image.load_img(img_path, target_size=img_size)
    img_array = np.expand_dims(image.img_to_array(img)/255.0, axis=0)
    
    preds = model.predict(img_array)
    predicted_class_idx = np.argmax(preds, axis=1)[0]
    
    if class_names is None:
        raise ValueError("class_names ne peut pas être None")
    
    predicted_class = class_names[predicted_class_idx]
    
    print(f"🔍 Image : {os.path.basename(img_path)}")
    print(f"🌿 Classe prédite : {predicted_class}")
    print(f"Confiance : {preds[0][predicted_class_idx]*100:.2f}%")
    
    return predicted_class, preds[0][predicted_class_idx]

# ========================================
# 🔹 Tracer courbes train/val accuracy et loss
# ========================================
def plot_training_history(history):
    plt.figure(figsize=(12,5))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# ========================================
# 🔹 Afficher quelques images d’un batch
# ========================================
def display_batch(images, labels, class_names, n=5):
    plt.figure(figsize=(15,5))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i].astype('uint8'))
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.show()
