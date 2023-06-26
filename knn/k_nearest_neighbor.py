# from collections import Counter
# import numpy as np
from collections import Counter
import glob
import os
import re
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from knn.cosine import Cosine
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tfidf.extraction_helper import Helper

# class KNN():
#     def majority_voting(self, cosine_similarity):
#         cosine_similarityList = list(cosine_similarity.values())
#         goals = list(cosine_similarity.keys())
#         nearest_labels = goals
#         label_mapping = {label: index for index,
#                          label in enumerate(set(goals))}
#         sorted_indices = np.argsort(cosine_similarityList)[::-1]
#         sorted_scores = np.array(cosine_similarityList)[sorted_indices]
#         sorted_labels = np.array(nearest_labels)[sorted_indices]
#         num_labels = len(label_mapping)
#         weighted_votes = np.zeros(num_labels)
#         for i, score in enumerate(sorted_scores):
#             label = sorted_labels[i]
#             index = label_mapping[label]
#             weighted_votes[index] += score
#         predicted_index = np.argmax(weighted_votes)
#         predicted_label = list(label_mapping.keys())[list(
#             label_mapping.values()).index(predicted_index)]

#         return predicted_label


#     def knn_rework(self)

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# documents = [
#     "In modern centralized information-based societies, information dissemination  has become one of the most critical social processes that enable people to develop  active  learning  initiatives  (Huang,  2021).  As  a  place  where  much  information  is  accumulated  and  handled,  schools  are  the  ideal  place  where  information  dissemination  is  practiced  and  streamlined  (Cope  &  Kalantzis,  2016).  Efficient  information dissemination in schools is crucial, such as improving education and  learning, promoting active participation and engagement with students and parents,  enhancing  school  administration  and  management,  and  ensuring  safety  and  emergency  communications  (Sadiku  et  al.,  2021).  However,  not  all  schools  can  achieve the best effect that effective information dissemination can provide. Schools  in the Philippines face several challenges with information dissemination due to their  diverse culture and unique geographical and educational landscape. Hence, it is  essential to use the current information infrastructure and technology, such as the  Internet of Things (IoT), to enhance information dissemination within schools in the  Philippines.   Schools utilize multiple applications and social networking sites to disseminate  information to their students, parents, and school members. Social media sites like  Facebook are the most popular and commonly used. Facebook is a social media  platform founded by Mark Zuckerberg in 2004. It enables users to create profiles, share  pictures, and connect with families and friends. It also allows users to create pages  and groups to get updated with news and content. Sites like Twitter have similar  features,  albeit  rarely  used  by  schools.  Some  applications  allow  their  users  to  exchange information, similar to forums. Among them, the most popular one that most ",
#     "One issue that many people nowadays face is that the job or school where they wish to work, or study is situated a long distance away from their homes. They must go from one island, city, or province to another, and many are unfamiliar with the region. Others possess real estate holdings that they want to rent out as a secondary source of income but have no idea how to advertise them properly. ARent, an online and mobile growth and economic",
#     "Education plays a crucial role in shaping individuals and societies. It is the key that unlocks doors to knowledge, opportunity, and personal growth. Education empowers individuals with the skills and knowledge needed to navigate through life's challenges and pursue their goals. Beyond acquiring subject-specific knowledge, education equips individuals with critical thinking abilities, problem-solving skills, and a broad perspective on the world. It enables people to understand diverse cultures, appreciate different viewpoints, and promotes tolerance and empathy. Moreover, education is not limited to formal institutions; it encompasses lifelong learning and continuous personal development. In essence, education is the foundation upon which individuals build fulfilling lives and contribute meaningfully to their communities.",
# ]

cons = Cosine()


class KNN():
    def knn_classifier(self, cosine_similarity, k):  # the value of K is 5
        cosine_similarityList = list(cosine_similarity.values())
        goals = list(cosine_similarity.keys())
        nearest_labels = goals[:k]
        nearest_scores = cosine_similarityList[:k]
        weighted_votes = Counter()
        for i, label in enumerate(nearest_labels):
            weighted_votes[label] += nearest_scores[i]
        predicted_label = weighted_votes.most_common(4)
        resultDictionary = dict((x, y) for x, y in predicted_label)
        return resultDictionary

    def preprocess_text(self, document):
        # Remove special characters and convert to lowercase
        document = re.sub(r'[^a-zA-Z\s]', '', document.lower())
        # Tokenize the document
        tokens = nltk.word_tokenize(document)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join the tokens back into a single string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def testing(self, data, n):
        exec = []
        start_time = time.time()
        trainingDocs = []
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
                 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
                 'Goal 13',
                 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
                 ]
        for goal in goals:
            trainingData = cons.extractAllPDF(goal)
            trainingDocs.append(trainingData)
        preprocessed_documents = [self.preprocess_text(
            document) for document in trainingDocs]
        vectorizer = TfidfVectorizer()

        # Fit and transform the preprocessed documents
        tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
        # Assuming you have a target variable indicating the UN SDG labels for each document
        target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16]  # Replace with the actual labels

        # Initialize the KNeighborsClassifier
        # Specify the number of neighbors
        knn_model = KNeighborsClassifier(n_neighbors=n)
        # Fit the model with the TF-IDF matrix and target labels
        knn_model.fit(tfidf_matrix, target_labels)
        # Preprocess the new document
        # new_document = """Plants are categorized into smaller groups according to their shared characteristics, which can be daunting given their complexity. While experts can quickly recognize familiar plants, identifying potentially harmful or toxic ones, particularly in medicine, can be challenging. Botanists possess the expertise to distinguish such plants, but with millions of species featuring similar parts (roots, stems, leaves), they must devise a system to classify them effectively. Living Green aims to expound botanical research through a Mobile Botanical Identifier mobile application for finding an unknown plant's captured or uploaded photo with a barter system. It is a mobile application that connects users with plant enthusiasts and plant experts to aid in identifying a plant name."""
        preprocessed_new_document = self.preprocess_text(data)
        # Transform the preprocessed new document using the fitted vectorizer
        new_tfidf = vectorizer.transform([preprocessed_new_document])

        # Use the trained KNN model to predict the UN SDG label
        predicted_label = knn_model.predict(new_tfidf)
        # arr_str = np.array2string(predicted_label)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # exec.append(execution_time)
#         cosine_similarities = cosine_similarity(new_tfidf, tfidf_matrix)

# # Generate random colors for each training document
#         training_colors = np.random.rand(len(trainingDocs))
#         new_color = 'red'  # Color for the new document

#         # Calculate the number of training documents and the index range
#         num_training_docs = len(trainingDocs)
#         training_indices = range(num_training_docs)

#         # Create an array of indices for the scatter plot
#         indices = np.concatenate([training_indices, [num_training_docs]])

#         # Repeat the cosine similarity score for each training document
#         cosine_scores = np.repeat(
#             cosine_similarities, num_training_docs, axis=1)

#         # Flatten the cosine scores and indices arrays
#         flattened_scores = cosine_scores.flatten()
#         flattened_indices = np.tile(indices, len(cosine_similarities))

#         # Plot the scatter plot with cosine similarity on the x-axis and document indices on the y-axis
#         plt.scatter(flattened_scores, flattened_indices,
#                     c=training_colors, label='Training Documents')
#         # Assuming cosine similarity of 1 for the new document
#         plt.scatter([1.0], [num_training_docs],
#                     c=new_color, label='New Document')

#         # Set the x-axis and y-axis labels
#         plt.xlabel('Cosine Similarity')
#         plt.ylabel('Document')

#         # Show the legend
#         plt.legend()

#         # Show the scatter plot
#         plt.show()
        print(predicted_label)
        return predicted_label

    def getFromPDF(self, filename):  # notused
        finalText = " "
        with pdfplumber.open('C:/Users/Dennis/Documents/COMICS/College/Test PDF/Test/Test Set' + + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                break
            extractFromPDF = ""
        return finalText

    def scatter_plot(self, training_tfidf_matrix, new_document_tfidf):

        # Apply PCA for dimensionality reduction to 2 dimensions
        pca = PCA(n_components=2)
        reduced_tfidf = pca.fit_transform(training_tfidf_matrix)

        # Separate the reduced data into training data and new document data
        # Exclude the last row (new document)
        reduced_training_tfidf = reduced_tfidf[:-1]
        # Last row (new document)
        reduced_new_document_tfidf = reduced_tfidf[-1]

        # Create a scatter plot for the training data
        plt.scatter(
            reduced_training_tfidf[:, 0], reduced_training_tfidf[:, 1], label='Training Data')

        # Plot the new document as a separate point with a different color or marker
        # plt.scatter(
        #     reduced_new_document_tfidf[0], reduced_new_document_tfidf[1], c='r', marker='x', label='New Document')

        # Add labels, legend, and other plot customizations
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('TF-IDF Visualization')
        plt.legend()
        plt.show()

    def automated_testing(self):
        print("Testing TFIDF-KNN ")
        list_results = []
        list_pdf = []
        helper = Helper()
        finale = {}
        directory = (glob.glob(
            "C:/Users/Dennis/Documents/COMICS/College/Test PDF/Test" + "/*.pdf"))
        extractedText, finalText, appendedData = " ", " ", " "
        for file in directory:
            file = file.replace("\\", "/")
            print("Testing File: ", file)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extractedText = page.extract_text()
                    finalText = finalText + extractedText
                string = os.path.basename(file)
                list_pdf.append(string)
                result = helper.main_logic(string)
                appendedData = result['appendedData']
                res = self.testing(appendedData, 10)
                list_results.append(res)
                finale[string] = res
        # print("Final List of Results: ", list_results)
        # print("Final List of Files: ", list_pdf)
        for i in finale:
            print(i, finale[i])
