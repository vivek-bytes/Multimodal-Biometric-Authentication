import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load your pre-trained models
ecg_model = tf.keras.models.load_model(r"C:\Users\aluvo\OneDrive\Desktop\model.h5")
face_model = tf.keras.models.load_model(r"C:\Users\aluvo\OneDrive\Desktop\vgg16_face.h5")

# Define paths to your datasets
ecg_data_path = r"C:\Users\aluvo\OneDrive\Desktop\Fussion\ECG"
face_data_path = r"C:\Users\aluvo\OneDrive\Desktop\Fussion\face"

# Create data generators for both datasets
datagen = ImageDataGenerator(rescale=1.0/255.0)

ecg_generator = datagen.flow_from_directory(
    ecg_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

face_generator = datagen.flow_from_directory(
    face_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get predictions for ECG data
ecg_predictions = ecg_model.predict(ecg_generator)
ecg_predicted_classes = np.argmax(ecg_predictions, axis=1)

# Get predictions for face data
face_predictions = face_model.predict(face_generator)
face_predicted_classes = np.argmax(face_predictions, axis=1)

# Get true labels
true_labels_ecg = ecg_generator.classes
true_labels_face = face_generator.classes

# Assuming the order of samples in both generators is the same
correct_predictions = 0
total_samples = len(ecg_predicted_classes)

# Initialize lists to store the results
true_labels = []
fused_predictions = []

for i in range(total_samples):
    # Use the ECG predictions as the primary and face predictions as secondary
    if ecg_predicted_classes[i] == face_predicted_classes[i]:
        fused_predictions.append(ecg_predicted_classes[i])
    else:
        # Choose one of the models or use a different strategy for disagreement
        fused_predictions.append(ecg_predicted_classes[i])  # or face_predicted_classes[i]

    true_labels.append(true_labels_ecg[i])  # Assuming true_labels_ecg and true_labels_face are the same

# Generate class names dynamically
num_classes = len(ecg_generator.class_indices)
class_names = [f'person_{i+1}' for i in range(num_classes)]

# Calculate confusion matrix
cm = confusion_matrix(true_labels, fused_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Fusion Model')
plt.show()

# Optionally, print a classification report for more details
print("\nClassification Report:")
print(classification_report(true_labels, fused_predictions, target_names=class_names))
