#🔹 Phase 1 (train.py) → entraînement de base :
#tu ajoutes des couches et tu entraînes uniquement celles-ci.

#🔹 Phase 2 (finetune.py) → fine-tuning :
#tu réouvres le modèle sauvegardé, tu dégèles quelques couches du bas de VGG16, et tu le réentraînes légèrement avec un learning_rate plus petit.


import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# === 1. Paths & Params ===
dataset_dir = '../data/Chili_Plant_Disease'
model_dir = '../models'
fine_tuned_model_path = os.path.join(model_dir, 'vgg16_finetuned_v2.h5')
history_path = os.path.join(model_dir, 'history_finetune.pkl')

train_dir = os.path.join(dataset_dir, 'train')
val_dir   = os.path.join(dataset_dir, 'val')

img_size = (224, 224)
batch_size = 32
fine_tune_epochs = 10
learning_rate = 1e-5  # très petit taux pour éviter de “casser” les poids appris

# === 2. Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# === 3. Load pre-trained model ===
model_path = os.path.join(model_dir, 'vgg16_finetuned.h5')
model = load_model(model_path)
print("✅ Model loaded successfully!")

# === 4. Unfreeze last convolutional blocks of VGG16 ===
# On ne veut pas tout défiger, juste les dernières couches profondes
fine_tune_at = 15  # par exemple, à partir de la 15e couche de VGG16

for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in model.layers[fine_tune_at:]:
    layer.trainable = True

print(f"ℹ️ Fine-tuning à partir de la couche {fine_tune_at}. Couches débloquées : {[l.name for l in model.layers[fine_tune_at:]]}")

# === 5. Recompile the model ===
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 6. Fine-tune ===
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=fine_tune_epochs
)

# === 7. Save new fine-tuned model ===
model.save(fine_tuned_model_path)
print("✅ Fine-tuned model saved successfully!")

# === 8. Save fine-tuning history ===
with open(history_path, 'wb') as f:
    pickle.dump(history_fine.history, f)
print("✅ Fine-tuning history saved!")
