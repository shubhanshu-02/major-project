from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation

def create_model():
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, (3,3), input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    # Block 4
    model.add(Conv2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.5))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    return model

# This is what you would have seen during training
model = create_model()
print("MODEL ARCHITECTURE YOU WOULD HAVE SEEN:")
print("=" * 50)
model.summary()

print(f"\nTOTAL LAYERS: {len(model.layers)}")
print(f"PARAMETERS: {model.count_params():,}")

print("\nKEY POINTS YOU WOULD HAVE KNOWN:")
print("- 8 Convolutional layers (2x32, 2x64, 2x128, 2x256 filters)")
print("- 4 MaxPooling layers for downsampling")
print("- BatchNormalization after each Conv2D")
print("- Progressive dropout (0.25 → 0.5)")
print("- 2 Dense layers (64 units each)")
print("- Output: 7 classes (emotions)")