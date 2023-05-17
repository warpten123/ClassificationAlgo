from collections import Counter
import numpy as np


class KNN():
    def majority_voting(self, cosine_similarity):
        cosine_similarityList = list(cosine_similarity.values())
        goals = list(cosine_similarity.keys())
        nearest_labels = goals
        label_mapping = {label: index for index,
                         label in enumerate(set(goals))}
        sorted_indices = np.argsort(cosine_similarityList)[::-1]
        sorted_scores = np.array(cosine_similarityList)[sorted_indices]
        sorted_labels = np.array(nearest_labels)[sorted_indices]
        num_labels = len(label_mapping)
        weighted_votes = np.zeros(num_labels)
        for i, score in enumerate(sorted_scores):
            label = sorted_labels[i]
            index = label_mapping[label]
            weighted_votes[index] += score
            print(weighted_votes[index])
        predicted_index = np.argmax(weighted_votes)
        predicted_label = list(label_mapping.keys())[list(
            label_mapping.values()).index(predicted_index)]

        return predicted_label

    def knn_classifier(self, cosine_similarity, k):
        cosine_similarityList = list(cosine_similarity.values())
        goals = list(cosine_similarity.keys())
        nearest_labels = goals[:k]
        nearest_scores = cosine_similarityList[:k]

        # Perform majority voting based on the weighted scores
        weighted_votes = Counter()
        for i, label in enumerate(nearest_labels):
            weighted_votes[label] += nearest_scores[i]
        print(weighted_votes)
        predicted_label = weighted_votes.most_common(4)
        resultDictionary = dict((x, y) for x, y in predicted_label)

        return resultDictionary
