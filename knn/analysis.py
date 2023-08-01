import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
cols = ["Model", "TFIDF with Cosine Similarity Scoring", "TFIDF Only", "TFIDF with KNN k = 1",
        "TFIDF with KNN k = 2", "TFIDF with KNN k = 3", "TFIDF with KNN k = 4", "TFIDF with KNN k = 5",
        "TFIDF with KNN k = 6", "TFIDF with KNN k = 10", "TFIDF with KNN k = 17"]
rows = [
    "Model", "Accuracy"]

# data = [
#     ["Average Classificaction Speed", 40.45, 39.71,
#         42.83, 42.89, 43.26, 43.11, 43.37, 44.33, 44.70, 43.37, 22],
#     # ["Testing Process Speed", 16.9, 16.5,
#     #  17.8, 17.7, 18.02, 17.96, 18.1, 18.47, 18.62, 17.38, 0],
#     # ["Testing Process", 26, 26, 26, 27, 37, 36, 28, 28, 58, 21, 22],

# ]
data = [

    ["Accuracy", 0.68, 0.52, 0.76, 0.48,
        0.28, 0.16, 0.12, 0.12, 0.04, 0.04, 0.0]
]

model_names = [row[0] for row in data]
data_values = np.array([row[1:] for row in data], dtype=float)

# Set the number of rows and columns
num_rows = len(rows) - 1
num_cols = len(cols) - 1

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x-axis range
x = np.arange(num_cols)

# Plot the data
width = 0.2  # Width of the bars
for i in range(num_rows):
    ax.bar(x + (i * width), data_values[i, :-1], width=width, label=rows[i+1])
    # Add text annotations
    font_props = FontProperties(weight='bold')
    for j, value in enumerate(data_values[i, :-1]):
        ax.text(x[j] + (i * width), value,
                f"{value:.2f}", ha="center", va="bottom", fontproperties=font_props)

# Set the x-axis ticks and labels
ax.set_xticks(x + ((num_rows - 1) * width) / 2)
labels = ax.set_xticklabels(cols[1:], rotation=45)
font_props = FontProperties(weight='bold')
for label in labels:
    label.set_font_properties(font_props)
# Set the y-axis label
ax.set_ylabel("Values")

# Set the title
ax.set_title("Accuracy Model Comparison")

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()


# data = ["Testing Process Speed", 16.9, 16.5,
#         17.8, 17.7, 18.02, 17.96, 18.1, 18.47, 18.62, 17.38, 0],

# model_names = [row[0] for row in data]
# data_values = np.array([row[1:] for row in data], dtype=float)

# # Set the number of rows and columns
# num_rows = len(rows) - 1
# num_cols = len(cols) - 1

# # Create the figure and axes
# fig, ax = plt.subplots(figsize=(10, 6))

# # Set the x-axis range
# x = np.arange(num_cols)

# # Plot the data
# for i in range(num_rows):
#     ax.plot(x, data_values[i, :-1], marker='o',
#             label=rows[i+1])

# # Set the x-axis ticks and labels
# ax.set_xticks(x)
# ax.set_xticklabels(cols[1:], rotation=45)

# # Set the y-axis label
# ax.set_ylabel("Accuracy Scores")

# # Set the title
# ax.set_title("Model Comparison for Accuracy")

# # Add a legend
# ax.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()
