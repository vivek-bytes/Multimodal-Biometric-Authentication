    

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

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

# Get probability scores for ECG data
ecg_probabilities = ecg_model.predict(ecg_generator)

# Get probability scores for face data
face_probabilities = face_model.predict(face_generator)

# Get true labels
true_labels = ecg_generator.classes  # Assuming true labels for ECG and face are the same

# Calculate fusion scores by averaging the probabilities
fused_probabilities = (ecg_probabilities + face_probabilities) / 2.0

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = ecg_probabilities.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels, fused_probabilities[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate macro-average ROC curve and AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["macro"], tpr["macro"], color='blue', lw=2, label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fusion Model')
plt.legend(loc="lower right")
plt.show()



