import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pickle

# === 1. Paths & Parameters ===
dataset_dir = 'data/Chili_Plant_Disease'  # <- change selon ton dataset
img_size = (224, 224)
batch_size = 32
epochs = 15
model_dir = 'models' 

train_dir = os.path.join(dataset_dir, 'train')
val_dir   = os.path.join(dataset_dir, 'val')
test_dir  = os.path.join(dataset_dir, 'test')

# === 2. Data Augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,      # normalise les pixels (de 0–255 → 0–1) pour un apprentissage plus stable.
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,  # retournement
    vertical_flip=True
)


# Pour la validation/test, on ne fait pas d’augmentation, juste la normalisation.
test_datagen = ImageDataGenerator(rescale=1./255)


#création des générateurs
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  #les étiquette selon les noms des sous-dossiers
)

val_gen = test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# === 3. Build Model ===  (VGG16 + couches personnalisées)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model.trainable = False  # freeze convolutional base


#ajout des couches personnalisées
x = GlobalAveragePooling2D()(base_model.output)  #convertit les cartes de caractéristiques en un vecteur
x = Dense(256, activation='relu')(x) # ajoute une couche fully connected de 256 neurones
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === 4. Compile Model ===
model.compile(
    optimizer=Adam(learning_rate=0.0001),  #Taux d’apprentissage 0.0001 : petit pour un entraînement stable.
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 5. Train Model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# === 6. Save Model ===
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'vgg16_finetuned.h5'))
print("✅ Model saved successfully!")

# === 7. Save Training History ===
with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Training history saved!")

