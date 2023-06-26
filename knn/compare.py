
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, accuracy_score
# expected = [8, 15, 4, 11, 3, 4, 9, 4, 8, 4, 1, 4,
#             3, 4, 4, 4, 16, 8, 8, 9, 8, 8, 8, 8, 8]
expected = [8, 15, 4, 9, 3, 4, 9, 4, 8, 4, 1,
            4, 3, 17, 4, 4, 16, 8, 8, 8, 8, 8, 8, 8, 11]


# actual_tfidf_cosine = [8, 15, 4, 16, 3, 4, 16, 16, 8, 4, 16,
#                        16, 3, 12, 4, 4, 16, 8, 8, 8, 8, 16, 8, 8, 8]
# actual_tfidf_only = [4, 15, 4, 4, 3, 8, 13, 4, 8,
#                      4, 4, 4, 9, 4, 4, 8, 4, 4, 8, 8, 8, 8, 12, 8, 4]
# actual_tfidf_knn_1 = [11, 15, 4, 9, 3, 4, 9, 4, 8,
#                       4, 4, 4, 9, 9, 4, 4, 16, 12, 8, 8, 8, 8, 8, 4, 11]
# actual_tidf_knn_2 = [4, 13, 1, 2, 3, 3, 3, 4, 8, 3,
#                      4, 4, 8, 9, 4, 4, 5, 9, 8, 8, 8, 8, 8, 4, 11]
# actual_tfidf_knn_3 = [4, 2, 1, 2, 3, 3, 3, 4, 8,
#                       3, 3, 3, 8, 2, 4, 3, 5, 8, 3, 4, 4, 8, 8, 4, 2]
# actual_tfidf_knn_4 = [4, 2, 1, 2, 3, 3, 3, 4, 8,
#                       3, 3, 3, 4, 2, 4, 3, 5, 4, 3, 4, 4, 4, 4, 2, 2]
# actual_tfidf_knn_5 = [4, 2, 1, 2, 2, 3, 3, 4, 4,
#                       3, 3, 3, 3, 1, 4, 3, 4, 4, 3, 4, 4, 4, 3, 2, 1]
actual_tfidf_knn_6 = [2, 2, 1, 2, 2, 3, 3, 4, 4,
                      2, 3, 3, 3, 1, 4, 3, 4, 2, 3, 4, 2, 1, 3, 2, 1]
labels_goals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# actual_tfidf_knn_17 = [1, 1, 1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def result(actual):
    matrix = confusion_matrix(actual, expected, labels=labels_goals)

    formatted_matrix = [
        '[' + ', '.join(str(value) for value in row) + '],' for row in matrix]

    # Print the formatted matrix
    for row in formatted_matrix:
        print(row)
    # Print the matrix with commas

    # precision = precision_score(actual, expected,
    #                             labels=labels_goals, average=None, zero_division=1)
    # recall = recall_score(actual, expected,
    #                       labels=labels_goals, average=None, zero_division=1)
    # f1 = f1_score(actual, expected, labels=labels_goals,
    #               average=None, zero_division=1)

    # micro_precision = precision_score(
    #     actual, expected, labels=labels_goals, average='micro', zero_division=1)
    # macro_precision = precision_score(
    #     actual, expected, labels=labels_goals, average='macro', zero_division=1)
    # weighted_precision = precision_score(
    #     actual, expected, labels=labels_goals, average='weighted', zero_division=1)

    report = classification_report(
        actual, expected, labels=labels_goals, target_names=[
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
        ], zero_division=1)
    accuracy = accuracy_score(actual, expected)
    report += f"\nAccuracy: {accuracy:.2f}"
    return report


def visualize_classification_report(report):
    lines = report.split('\n')

    # Extract class names and metrics from report
    classes = []
    plot_data = []
    for line in lines[2:(len(lines)-5)]:
        row_data = line.split()
        if len(row_data) > 0 and row_data[0].isdigit():
            classes.append(row_data[0])
            plot_data.append([float(x) for x in row_data[1:]])

    # Remove classes with zero values for precision, recall, and F1-score
    non_zero_classes = []
    non_zero_plot_data = []
    for i, data in enumerate(plot_data):
        if not np.allclose(data, 0):
            non_zero_classes.append(classes[i])
            non_zero_plot_data.append(data)

    # Convert plot data to a numpy array
    plot_data_array = np.array(non_zero_plot_data)

    # Check if the plot data array is singular
    if np.linalg.matrix_rank(plot_data_array) < plot_data_array.shape[1]:
        print("Unable to visualize the classification report. The plot data matrix is singular.")
        return

    # Plot precision, recall, and F1-score for each non-zero class
    fig, ax = plt.subplots(figsize=(8, len(non_zero_classes) * 0.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, len(non_zero_classes) + 0.5)
    ax.set_xlabel('Metric')
    ax.set_title('Classification Report')

    y_ticks = range(len(non_zero_classes))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(non_zero_classes)

    # Change or extend colors as needed
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    for i, metric in enumerate(metrics):
        for j in range(len(non_zero_classes)):
            value = non_zero_plot_data[j][i]
            color = colors[j % len(colors)]
            ax.text(value, j, f'{value:.2f}', color=color, va='center')
            ax.barh(j, value, color=color, alpha=0.6)

    plt.tight_layout()
    plt.show()

# result(actual_tfidf_cosine)


report = result(actual_tfidf_knn_6)
print("\n", report)
# visualize_classification_report(report)
