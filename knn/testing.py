import glob
import os
import time
import pdfplumber
from knn.cosine import Cosine
from knn.k_nearest_neighbor import KNN
from tfidf.extraction_helper import Helper


class Testing():
    # C:/Users/Dennis/Documents/COMICS/College/Test PDF/Test
    def getFromPDF(self, filename):  # notused
        finalText = " "
        with pdfplumber.open('C:/Users/Dennis/Documents/COMICS/College/Test PDF/Test/Test Set' + + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                break
            extractFromPDF = ""
        return finalText

    def extractAllPDF(self):
        start_time = time.time()
        helper = Helper()
        cosine = Cosine()
        knn = KNN()
        listOfPredicted = []
        directory = (glob.glob(
            "C:/Users/Dennis/Documents/COMICS/College/Test PDF/Test" + "/*.pdf"))
        extractedText, finalText, appendedData = " ", " ", " "
        for file in directory:
            file = file.replace("\\", "/")
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extractedText = page.extract_text()
                    finalText = finalText + extractedText
                string = os.path.basename(file)
                result = helper.main_logic(string)
                appendedData = result['appendedData']
                data = cosine.classifyResearch(appendedData, True)
                finalRes = knn.knn_classifier(data, 5)
                predict = self.accuracyTesting(finalRes, string)
                listOfPredicted.append(predict)
        print(listOfPredicted)
        count = listOfPredicted.count(True)
        print("SDG Classifier Accuracy: ", round(
            count / len(listOfPredicted), 2) * 100, "%")
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        return finalRes

    def accuracyTesting(self, finalRes, fileName):
        goals = ["Goal 1: No Poverty", "Goal 2: Zero Hunger", "Goal 3: Good Health and Well-Being", "Goal 4: Quality Education",
                 "Goal 5: Gender Equality", "Goal 6: Clean Water and Sanitation", "Goal 7: Affordable and Clean Energy",
                 "Goal 8: Decent Work and Economic Growth", "Goal 9: Industry, Innovation, and Infrastrucuture",
                 "Goal 10: Reduced Inequalities", "Goal 11: Sustainable Cities and Communities",
                 "Goal 12: Responsible Consumption and Production", "Goal 13: Climate Action",
                 "Goal 14: Life Below Water", "Goal 15: Life on Land", "Goal 16: Peace, Justice and Strong Institutions",
                 "Goal 17: Partnership for the Goals"
                 ]

        correct = False
        correctLabels = {}
        correctLabels['Test Set 2.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 1: No Poverty',
                                           'Goal 9: Industry, Innovation, and Infrastrucuture', 'Goal 16: Peace, Justice and Strong Institutions']
        correctLabels['Test Set 3.pdf'] = ['Goal 15: Life on Land', 'Goal 14: Life Below Water',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 4.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 2: Zero Hunger']
        correctLabels['Test Set 5.pdf'] = ['Goal 11: Sustainable Cities and Communities', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 6.pdf'] = ['Goal 3: Good Health and Well-Being', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 7.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 8.pdf'] = ['Goal 9: Industry, Innovation, and Infrastrucuture', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth'],
        correctLabels['Test Set 9.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                           'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 10.pdf'] = ['Goal 3: Good Health and Well-Being', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 11.pdf'] = ['Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 12.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 11: Sustainable Cities and Communities']
        correctLabels['Test Set 13.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 11: Sustainable Cities and Communities']
        correctLabels['Test Set 14.pdf'] = ['Goal 15: Life on Land', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 14: Life Below Water']
        correctLabels['Test Set 15.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 11: Sustainable Cities and Communities']
        correctLabels['Test Set 16.pdf'] = ['Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 17.pdf'] = ['Goal 11: Sustainable Cities and Communities', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 18.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture']
        correctLabels['Test Set 19.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 16: Peace, Justice and Strong Institutions']
        correctLabels['Test Set 20.pdf'] = ['Goal 1: No Poverty', 'Goal 4: Quality Education',
                                            'Goal 8: Decent Work and Economic Growth', 'Goal 2: Zero Hunger']
        correctLabels['Test Set 21.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 16: Peace, Justice and Strong Institutions',
                                            'Goal 3: Good Health and Well-Being', 'Goal 2: Zero Hunger']
        correctLabels['Test Set 22.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture',
                                            'Goal 3: Good Health and Well-Being']
        correctLabels['Test Set 23.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture',
                                            'Goal 3: Good Health and Well-Being']
        correctLabels['Test Set 24.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture',
                                            'Goal 3: Good Health and Well-Being']
        correctLabels['Test Set 25.pdf'] = ['Goal 8: Decent Work and Economic Growth', 'Goal 9: Industry, Innovation, and Infrastrucuture',
                                            'Goal 3: Good Health and Well-Being']

        if (fileName in correctLabels):
            keys = list(finalRes.keys())
            keys2 = list(correctLabels[fileName])
            word_counts = sum(1 for word in keys if word in keys2)
            if (word_counts >= 2):
                correct = True
            else:
                correct = False
            print(fileName, "\nExpected Labels: ",
                  correctLabels[fileName], "\nActual Results: ", keys, "\nResult: ", correct)
        return correct
