from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 1. IMPROVED MODEL ARCHITECTURE
def create_improved_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.5),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    return model

# 2. ENHANCED DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# 3. CLASS WEIGHT CALCULATION
def calculate_class_weights():
    train_generator = train_datagen.flow_from_directory(
        'images/train/',
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    return dict(enumerate(class_weights))

# 4. TRAINING IMPROVEMENTS
model = create_improved_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Train with class weights
class_weights = calculate_class_weights()
print("Class weights:", class_weights)

# Training code (uncomment to train)
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=validation_generator,
#     class_weight=class_weights,
#     callbacks=callbacks
# )