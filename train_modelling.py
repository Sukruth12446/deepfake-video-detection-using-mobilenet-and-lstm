import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
# Configuration parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 5
DATASET_PATH = '/kaggle/input/deepfake-and-real-images/dataset'
# %%
# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# %%
# Set up data generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# %%
def create_mobilenet_lstm_model():
    # Input layer
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Load MobileNet without top layers
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_tensor=input_layer
    )

    # Freeze the MobileNet layers
    base_model.trainable = False

    # Get the output from MobileNet
    x = base_model.output

    # Add Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Reshape for LSTM (treating features as temporal sequence)
    x = Reshape((1, -1))(x)  # Reshape to (1, feature_size)

    # Add LSTM layers
    x = LSTM(512, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = LSTM(256)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# %%
# Create and compile the model
model = create_mobilenet_lstm_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()


# %%
# Define callbacks
class DetailedProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Training Accuracy:     {logs['accuracy']:.4f}")
        print(f"Validation Accuracy:   {logs['val_accuracy']:.4f}")
        print(f"Training Loss:         {logs['loss']:.4f}")
        print(f"Validation Loss:       {logs['val_loss']:.4f}")
        print(f"Accuracy Gap:          {(logs['accuracy'] - logs['val_accuracy']):.4f}")
        print(f"Learning Rate:         {self.model.optimizer.learning_rate.numpy():.2e}")
        print("-" * 50)


# %%
# Define callbacks with the new progress monitor
callbacks = [
    DetailedProgressCallback(),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_mobilenet_lstm_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]
# %%
# Modified training with validation_steps specified
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    verbose=1
)


# %%
# Modified training history plot to show more detail
def plot_detailed_history(history):
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'bo-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final metrics summary
    print("\nTraining Summary:")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy:  {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best Training Loss: {min(history.history['loss']):.4f}")
    print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")


# After training, plot the detailed history
plot_detailed_history(history)
# %%
# Evaluate model on test set with more detailed output
print("\nEvaluating on Test Set:")
test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)
print(f"\nFinal Test Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save the model
model.save('mobilenet_lstm_deepfake_model.keras')
print("Model saved successfully!")


# %%
# Function for making predictions
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return prediction


# %%
# Generate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def evaluate_model_performance():
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))


# Evaluate model performance
evaluate_model_performance()


# %%
# Optional: Fine-tune MobileNet layers after initial training
def fine_tune_model():
    # Unfreeze some layers of MobileNet
    for layer in model.layers[0].layers[-30:]:  # Unfreeze last 30 layers
        layer.trainable = True

    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train for a few more epochs
    history_fine_tune = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    return history_fine_tune


# %%
import os
import json
import datetime
from tensorflow.keras.models import save_model
import numpy as np


def save_model_artifacts(model, history, test_results, output_dir='/kaggle/working/model_output'):
    """
    Save model and all associated artifacts to the specified output directory.

    Args:
        model: Trained Keras model
        history: Training history object
        test_results: Dictionary containing test metrics
        output_dir: Directory to save all outputs
    """
    # Create timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create main output directory
    model_dir = os.path.join(output_dir, f'mobilenet_lstm_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    try:
        # 1. Save the model
        model_path = os.path.join(model_dir, 'mobilenet_lstm_model.keras')
        save_model(model, model_path, save_format='keras_v3')

        # 2. Save model weights separately
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        model.save_weights(weights_path)

        # 3. Save model architecture as JSON
        architecture_path = os.path.join(model_dir, 'model_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())

        # 4. Save training history
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }

        history_path = os.path.join(model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)

        # 5. Save test results and model metrics
        metrics = {
            'test_accuracy': float(test_results[1]),
            'test_loss': float(test_results[0]),
            'final_training_accuracy': float(history.history['accuracy'][-1]),
            'final_validation_accuracy': float(history.history['val_accuracy'][-1]),
            'best_validation_accuracy': float(max(history.history['val_accuracy'])),
            'total_epochs': len(history.history['accuracy']),
            'timestamp': timestamp
        }

        metrics_path = os.path.join(model_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # 6. Save model summary
        summary_path = os.path.join(model_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        print(f"\nModel artifacts successfully saved to: {model_dir}")
        print("\nSaved files:")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"- {file}: {size_mb:.2f} MB")

    except Exception as e:
        print(f"Error saving model artifacts: {str(e)}")
        raise


# Function to load and verify saved model
def verify_saved_model(model_dir):
    """
    Load and verify saved model artifacts.

    Args:
        model_dir: Directory containing saved model artifacts
    """
    try:
        # Load model
        loaded_model = tf.keras.models.load_model(
            os.path.join(model_dir, 'mobilenet_lstm_model.keras')
        )

        # Load metrics
        with open(os.path.join(model_dir, 'model_metrics.json'), 'r') as f:
            loaded_metrics = json.load(f)

        print("\nModel verification:")
        print("- Model loaded successfully")
        print(f"- Test accuracy: {loaded_metrics['test_accuracy']:.4f}")
        print(f"- Best validation accuracy: {loaded_metrics['best_validation_accuracy']:.4f}")

        return loaded_model, loaded_metrics

    except Exception as e:
        print(f"Error verifying saved model: {str(e)}")
        raise


# Usage example:
def save_and_verify_model(model, history, test_generator):
    # Get test results
    test_results = model.evaluate(test_generator)

    # Save all model artifacts
    save_model_artifacts(model, history, test_results)

    # Get the directory where model was saved
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join('/kaggle/working/model_output', f'mobilenet_lstm_{timestamp}')

    # Verify saved model
    loaded_model, loaded_metrics = verify_saved_model(model_dir)

    return model_dir