"""
Emotion Detection Model Training Script
========================================
This script trains a custom CNN model for facial emotion recognition.
Dataset: FER-2013 (Facial Expression Recognition)

Requirements:
pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib --break-system-packages
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class EmotionCNN:
    """Custom CNN architecture for emotion detection"""
    
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build the CNN architecture"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_callbacks(self, model_path='best_emotion_model.h5'):
        """Define training callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks


def load_fer2013_data(csv_path):
    """
    Load FER-2013 dataset from CSV file
    Download from: https://www.kaggle.com/datasets/msambare/fer2013
    """
    df = pd.read_csv(csv_path)
    
    # Extract pixels and labels
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()
    
    # Convert to numpy arrays
    X = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.array(face).reshape(48, 48)
        X.append(face)
    
    X = np.array(X)
    y = np.array(emotions)
    
    # Normalize pixel values
    X = X / 255.0
    
    # Reshape for CNN input
    X = X.reshape(-1, 48, 48, 1)
    
    # Convert labels to categorical
    y = keras.utils.to_categorical(y, num_classes=7)
    
    return X, y


def create_data_augmentation():
    """Create image data augmentation generator"""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    return datagen


def train_model(csv_path, epochs=100, batch_size=64):
    """Main training function"""
    
    print("Loading data...")
    X, y = load_fer2013_data(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train.argmax(axis=1)
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\nBuilding model...")
    emotion_cnn = EmotionCNN()
    emotion_cnn.compile_model()
    
    print(emotion_cnn.model.summary())
    
    # Data augmentation
    datagen = create_data_augmentation()
    datagen.fit(X_train)
    
    # Training
    print("\nStarting training...")
    history = emotion_cnn.model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=emotion_cnn.get_callbacks(),
        verbose=1
    )
    
    # Evaluation
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = emotion_cnn.model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    emotion_cnn.model.save('emotion_detection_final.h5')
    print("\nModel saved as 'emotion_detection_final.h5'")
    
    return emotion_cnn.model, history


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")


class RealTimeEmotionDetector:
    """Real-time emotion detection using webcam"""
    
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotions = EMOTIONS
        
    def detect_emotion(self, frame):
        """Detect emotion from a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = face_roi.reshape(1, 48, 48, 1)
            
            # Predict emotion
            predictions = self.model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = self.emotions[emotion_idx]
            confidence = predictions[emotion_idx]
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': dict(zip(self.emotions, predictions))
            })
            
        return results
    
    def run_webcam(self):
        """Run real-time detection on webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect emotions
            results = self.detect_emotion(frame)
            
            # Draw results
            for result in results:
                x, y, w, h = result['bbox']
                emotion = result['emotion']
                confidence = result['confidence']
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw emotion label
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        csv_path = sys.argv[2] if len(sys.argv) > 2 else 'fer2013.csv'
        train_model(csv_path)
    elif len(sys.argv) > 1 and sys.argv[1] == 'detect':
        # Detection mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_emotion_model.h5'
        detector = RealTimeEmotionDetector(model_path)
        detector.run_webcam()
    else:
        print("Usage:")
        print("  Training: python emotion_model.py train [csv_path]")
        print("  Detection: python emotion_model.py detect [model_path]")
