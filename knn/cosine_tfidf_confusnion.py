import numpy as np
import matplotlib.pyplot as plt


# confusion_matrix = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 17
#     # 1 2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
# ])

# TFIDF ONLY
confusion_matrix = np.array([
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


# TFIDF COSINE
# confusion_matrix = np.array([

#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 17
# ])

# Define SDG goals' names
sdg_goals = [
    "Goal 1",
    "Goal 2",
    "Goal 3",
    "Goal 4",
    "Goal 5",
    "Goal 6",
    "Goal 7",
    "Goal 8",
    "Goal 9",
    "Goal 10",
    "Goal 11",
    "Goal 12",
    "Goal 13",
    "Goal 14",
    "Goal 15",
    "Goal 16",
    "Goal 17",
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 5))

# Create heatmap
heatmap = ax.imshow(confusion_matrix, cmap='Blues')

# Set title and labels
ax.set_title("Confusion Matrix - TDIDF ONLY")
ax.set_xlabel("Expected Goals")
ax.set_ylabel("Actual Goals")

# Set tick labels
ax.set_xticks(np.arange(len(sdg_goals)))
ax.set_yticks(np.arange(len(sdg_goals)))
ax.set_xticklabels(sdg_goals, rotation=90)
ax.set_yticklabels(sdg_goals)

# Set threshold for text color
threshold = confusion_matrix.max() / 2

# Iterate over data and create text annotations
for i in range(len(sdg_goals)):
    for j in range(len(sdg_goals)):
        text_color = 'white' if confusion_matrix[i, j] > threshold else 'black'
        ax.text(j, i, confusion_matrix[i, j],
                ha='center', va='center', color=text_color)

# Add colorbar
cbar = plt.colorbar(heatmap)

# Show the plot
plt.tight_layout()
plt.show()
# confusion_matrix[np.isnan(confusion_matrix)] = 0
# accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
# precision = np.diag(confusion_matrix) / np.where(np.sum(confusion_matrix,
#                                                         axis=0) == 0, 1, np.sum(confusion_matrix, axis=0))
# recall = np.diag(confusion_matrix) / np.where(np.sum(confusion_matrix,
#                                                      axis=1) == 0, 1, np.sum(confusion_matrix, axis=1))
# f1_score = 2 * (precision * recall) / np.where(precision +
#                                                recall == 0, 1, precision + recall)
# print(accuracy, "\n")
# print("PRECISION: \n", precision, "\n")
# print("RECALL: \n", recall, "\n")
# print("F1 SCORE: \n", f1_score, "\n")

# tp = np.sum(np.diag(confusion_matrix))
# fp = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
# fn = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)

# # Calculate TN for the entire confusion matrix
# total_sum = np.sum(confusion_matrix)
# tn = (tp + np.sum(fp) + np.sum(fn)) - total_sum
# # import numpy as np

# # Example confusion matrix (17x17) with sample values
# confusion_matrix = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 17
# ])


# num_classes = confusion_matrix.shape[0]

# # Initialize TP, TN, FP, FN arrays
# tp = np.zeros(num_classes)
# tn = np.zeros(num_classes)
# fp = np.zeros(num_classes)
# fn = np.zeros(num_classes)

# # Calculate TP, TN, FP, FN for each class
# for i in range(num_classes):
#     tp[i] = confusion_matrix[i, i]
#     tn[i] = np.sum(confusion_matrix) - np.sum(confusion_matrix[i]
#                                               ) - np.sum(confusion_matrix[:, i]) + tp[i]
#     fp[i] = np.sum(confusion_matrix[:, i]) - tp[i]
#     fn[i] = np.sum(confusion_matrix[i]) - tp[i]

# print("True Positives (TP):", tp)
# print("True Negatives (TN):", tn)
# print("False Positives (FP):", fp)
# print("False Negatives (FN):", fn)

# import numpy as np

# # Confusion matrix
# confusion_matrix = np.array([

#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 17
#     # ])
# ])

# # Calculate TP, TN, FP, FN for each goal
# goals = range(1, 18)
# tp, tn, fp, fn = [], [], [], []
# for i in goals:
#     tp.append(confusion_matrix[i-1, i-1])
#     tn.append(np.sum(confusion_matrix) - np.sum(confusion_matrix[i-1]) - np.sum(
#         confusion_matrix[:, i-1]) + confusion_matrix[i-1, i-1])
#     fp.append(np.sum(confusion_matrix[:, i-1]) - confusion_matrix[i-1, i-1])
#     fn.append(np.sum(confusion_matrix[i-1]) - confusion_matrix[i-1, i-1])

# # Calculate precision, recall, and F1 score for each goal
# precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) >
#              0 else 0 for i in range(len(goals))]
# recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) >
#           0 else 0 for i in range(len(goals))]
# f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
#             if (precision[i] + recall[i]) > 0 else 0 for i in range(len(goals))]

# # Calculate overall metrics
# overall_precision = np.mean(precision)
# overall_recall = np.mean(recall)
# overall_f1_score = np.mean(f1_score)
# overall_accuracy = (np.sum(tp) + np.sum(fn)) / 25 * 100

# # Print TP, TN, FP, FN for each goal
# for i in goals:
#     print(f"Goal {i}: TP={tp[i-1]}, TN={tn[i-1]}, FP={fp[i-1]}, FN={fn[i-1]}")

# # Print precision, recall, F1 score, and accuracy for each goal
# for i in goals:
#     print(
#         f"Goal {i}: Precision={precision[i-1]:.3f}, Recall={recall[i-1]:.3f}, F1 Score={f1_score[i-1]:.3f}")

# # Print overall metrics
# print(f"Overall Precision: {overall_precision:.3f}")
# print(f"Overall Recall: {overall_recall:.3f}")
# print(f"Overall F1 Score: {overall_f1_score:.3f}")
# print(f"Overall Accuracy: {overall_accuracy:.3f}")
# import matplotlib.pyplot as plt

# Define the goals and corresponding metrics
# goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5', 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10',
#          'Goal 11', 'Goal 12', 'Goal 13', 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17']
# # precision = [0.000, 0.000, 1.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000,
# #              0.143, 0.000]
# # recall = [0.000, 0.000, 1.000, 0.833, 0.000, 0.000, 0.000, 0.889, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000,
# #           1.000, 0.000]
# # f1_score = [0.000, 0.000, 1.000, 0.909, 0.000, 0.000, 0.000, 0.941, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000,
# #             0.250, 0.000]
# tp = [0, 0, 2, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 1, 1, 0]

# fp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0]
# fn = [3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0]

# # Create a table with the metrics
# # table_data = [precision, recall, f1_score]
# # row_labels = ['Precision', 'Recall', 'F1 Score']

# # Create a table with the metrics
# table_data = [tp,  fp]
# table_rows = ['TP', 'FP']

# fig, ax = plt.subplots()
# ax.axis('off')
# table = ax.table(cellText=table_data, colLabels=goals, rowLabels=table_rows,
#                  loc='center', cellLoc='center')

# table.scale(1, 2)  # Adjust table size if needed
# table.auto_set_font_size(False)

# # Set font properties for the table cells
# for i in range(len(table_rows)):
#     for j in range(len(goals)):
#         cell = table.get_celld()[(i+1, j)]


# plt.show()

# # Define the matrix
# matrix = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 17
# ])

# Add a column and row of zeros for the missing class
# matrix = np.insert(matrix, matrix.shape[1], 0, axis=1)
# matrix = np.insert(matrix, matrix.shape[0], 0, axis=0)

# # Calculate TP, FP, TN, FN for each class
# tp = np.diag(matrix)
# fp = np.sum(matrix, axis=0) - tp
# fn = np.sum(matrix, axis=1) - tp
# tn = np.sum(matrix) - (tp + fp + fn)

# # Replace NaN values with zero
# tp = np.nan_to_num(tp)
# fp = np.nan_to_num(fp)
# fn = np.nan_to_num(fn)
# tn = np.nan_to_num(tn)

# # Calculate Precision, Recall, and F1 score for each class
# precision = np.where((tp + fp) != 0, tp / (tp + fp), 0)
# recall = np.where((tp + fn) != 0, tp / (tp + fn), 0)
# f1_score = np.where((precision + recall) != 0, 2 *
#                     (precision * recall) / (precision + recall), 0)

# # Check for zero denominators and set overall metrics to 0 if necessary
# if np.isnan(precision).all() or np.isnan(recall).all() or np.isnan(f1_score).all():
#     overall_precision = 0
#     overall_recall = 0
#     overall_f1_score = 0
# else:
#     # Calculate overall Precision, Recall, and F1 score
#     overall_precision = np.mean(precision)
#     overall_recall = np.mean(recall)
#     overall_f1_score = np.mean(f1_score)

# # Display the results
# for i in range(len(tp)):
#     class_num = i + 1
#     print(f"Class {class_num}:")
#     print(f"TP: {tp[i]}, FP: {fp[i]}, TN: {tn[i]}, FN: {fn[i]}")
#     print(
#         f"Precision: {precision[i]}, Recall: {recall[i]}, F1 Score: {f1_score[i]}\n")

# print(f"Overall Precision: {overall_precision}")
# print(f"Overall Recall: {overall_recall}")
# print(f"Overall F1 Score: {overall_f1_score}")
# total_tp = np.sum(tp)
# total_fp = np.sum(fp)
# total_tn = np.sum(tn)
# total_fn = np.sum(fn)

# print(f"Total TP: {total_tp}")
# print(f"Total FP: {total_fp}")
# print(f"Total TN: {total_tn}")
# print(f"Total FN: {total_fn}")

# Add a row of zeros for the missing row
# matrix = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],  # 1
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 17
# ])

# # Calculate TP, FP, TN, FN for each class
# TP = np.diag(matrix)
# FP = np.sum(matrix, axis=0) - TP
# FN = np.sum(matrix, axis=1) - TP
# TN = np.sum(matrix) - (TP + FP + FN)
# print(f"Total TP: {np.sum(TP)}")
# print(f"Total FP: {np.sum(FP)}")
# print(f"Total TN: {np.sum(TN)}")
# print(f"Total FN: {np.sum(FN)}")
# # Calculate precision, recall, and F1 score for each class
# denominator = TP + FP
# precision = np.where(denominator == 0, 0, TP / denominator)
# recall = np.where(TP + FN == 0, 0, TP / (TP + FN))
# f1_score = np.where((precision + recall) == 0, 0, 2 *
#                     (precision * recall) / (precision + recall))
# accuracy_per_class = np.diag(matrix) / np.sum(matrix, axis=1)
# accuracy_per_class[np.isnan(accuracy_per_class)] = 0
# average_accuracy = np.nanmean(accuracy_per_class)
# print(accuracy_per_class)
# non_zero_indices = np.nonzero(TP)  # Get the indices of non-zero TP values
# average_precision = np.mean(precision[non_zero_indices])
# average_recall = np.mean(recall[non_zero_indices])
# average_f1_score = np.mean(f1_score[non_zero_indices])

# print("Average Precision: ", average_precision)
# print("Average Recall: ", average_recall)
# print("Average F1 Score:", average_f1_score)
# # Create the tables
# class_labels = range(1, matrix.shape[0] + 1)

# table1_data = np.array([TP, FP, TN, FN]).T
# table1_columns = ["TP", "FP", "TN", "FN"]
# table1_rows = [f"Goal {label}" for label in class_labels]

# table2_data = np.array([precision, recall, f1_score]).T
# table2_columns = ["Precision", "Recall", "F1 Score"]
# table2_rows = [f"Goal1 {label}" for label in class_labels]

# # Plot the tables
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ax1.axis("off")
# ax1.table(cellText=table1_data, colLabels=table1_columns,
#           rowLabels=table1_rows, loc="center")
# ax1.set_title("Confusion Matrix")

# ax2.axis("off")
# ax2.table(cellText=table2_data, colLabels=table2_columns,
#           rowLabels=table2_rows, loc="center")
# ax2.set_title("Metrics")

# plt.tight_layout()
# plt.show()
