
import numpy as np
import matplotlib.pyplot as plt
# TFIDF ONLY
# matrix = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [1, 0, 0, 6, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 2, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 17
#     # 1 2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
# ])
# TFIDF + KNN
matrix = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2
    [2, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    [0, 0, 0, 3, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 17
    # 1 2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
])
num_cols = matrix.shape[1]
print(num_cols)
# Calculate TP, FP, TN, FN for each class
TP = np.diag(matrix)
FP = np.sum(matrix, axis=0) - TP
FN = np.sum(matrix, axis=1) - TP
TN = np.sum(matrix) - (TP + FP + FN)
print(f"Total TP: {np.sum(TP)}")
print(f"Total FP: {np.sum(FP)}")
print(f"Total TN: {np.sum(TN)}")
print(f"Total FN: {np.sum(FN)}")
# Calculate precision, recall, and F1 score for each class
denominator = TP + FP
precision = np.where(denominator == 0, 0, TP / denominator)
recall = np.where(TP + FN == 0, 0, TP / (TP + FN))
f1_score = np.where((precision + recall) == 0, 0, 2 *
                    (precision * recall) / (precision + recall))
accuracy_per_class = np.diag(matrix) / np.sum(matrix, axis=1)
accuracy_per_class[np.isnan(accuracy_per_class)] = 0
average_accuracy = np.nanmean(accuracy_per_class)
print(accuracy_per_class)
non_zero_indices = np.nonzero(TP)  # Get the indices of non-zero TP values
average_precision = np.mean(precision[non_zero_indices])
average_recall = np.mean(recall[non_zero_indices])
average_f1_score = np.mean(f1_score[non_zero_indices])
non_zero_values = accuracy_per_class[accuracy_per_class != 0]
accuracy = np.mean(non_zero_values)
print("Accuracy Of TFIDF+KNN with N=3: ", accuracy * 100)
print("Average Precision: ", average_precision)
print("Average Recall: ", average_recall)
print("Average F1 Score:", average_f1_score)

# Create the tables
class_labels = range(1, matrix.shape[0] + 1)

table1_data = np.array([TP, FP, TN, FN]).T
table1_columns = ["TP", "FP", "TN", "FN"]
table1_rows = [f"Goal {label}" for label in class_labels]

table2_data = np.array([precision, recall, f1_score]).T
table2_columns = ["Precision", "Recall", "F1 Score"]
table2_rows = [f"Goal1 {label}" for label in class_labels]

# Plot the tables
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.axis("off")
ax1.table(cellText=table1_data, colLabels=table1_columns,
          rowLabels=table1_rows, loc="center")
ax1.set_title("Confusion Matrix")

ax2.axis("off")
ax2.table(cellText=table2_data, colLabels=table2_columns,
          rowLabels=table2_rows, loc="center")
ax2.set_title("Metrics")

plt.tight_layout()
plt.show()
