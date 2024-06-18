#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = label # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
acc = accuracy_score(y_true, y_pred)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Calculate Balanced Accuracy
bacc = balanced_accuracy_score(y_true, y_pred)

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass

# Calculate Precision for multiclass
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate Recall for multiclass
recall = recall_score(y_true, y_pred, average='weighted')



# # Calculate AUC (Area Under the Curve)
# # roc_auc = roc_auc_score(y_true, y_pred, average='weighted')

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# # Calculate Geometric Mean from the confusion matrix
# # tn, fp, fn, tp, = conf_matrix.ravel()
# # g_mean = (tp / (tp + fn)) * (tn / (tn + fp))**0.5

# # Print or use these metrics as needed
print("Accuracy:", acc)
print("Cohen's Kappa:", kappa)
print("Matthews Correlation Coefficient:", mcc)
print("Balanced Accuracy:", bacc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
# print("AUC (Area Under the Curve):", roc_auc)
# print("Geometric Mean:", g_mean)
from sklearn.metrics import roc_auc_score

# Assuming you have true labels and predicted probabilities for each class
y_true = label  # Replace with your true labels
y_prob = global_model.predict(test)  # Replace with your predicted probabilities

# # Calculate AUC for multiclass classification
auc = roc_auc_score(y_true, y_prob, average='weighted')

# # Print or use the AUC value as needed
print("AUC (Area Under the Curve):", auc)
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming you have true labels and predicted labels for multiclass classification
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)  # Replace with your predicted labels

# Convert true and predicted labels to class labels (not one-hot encoded)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate G-Mean for each class
class_g_means = []
for class_label in range(5):  # Replace num_classes with the number of classes
    # Create a binary confusion matrix for the current class
    true_class = (y_true == class_label)
    pred_class = (y_pred == class_label)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()

    # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate G-Mean for the current class
    g_mean = np.sqrt(sensitivity * specificity)

    class_g_means.append(g_mean)

# Calculate the overall G-Mean (geometric mean of class G-Means)
overall_g_mean = np.prod(class_g_means) ** (1 / len(class_g_means))

# Print or use the overall G-Mean as needed
print("Overall G-Mean:", overall_g_mean)

