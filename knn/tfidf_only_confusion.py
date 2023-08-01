import glob
import os
import time

import pdfplumber
from knn.cosine import Cosine
from knn.k_nearest_neighbor import KNN
from tfidf.extraction_helper import Helper


cons = Cosine()
documents = [
    "In modern centralized information-based societies, information dissemination  has become one of the most critical social processes that enable people to develop  active  learning  initiatives  (Huang,  2021).  As  a  place  where  much  information  is  accumulated  and  handled,  schools  are  the  ideal  place  where  information  dissemination  is  practiced  and  streamlined  (Cope  &  Kalantzis,  2016).  Efficient  information dissemination in schools is crucial, such as improving education and  learning, promoting active participation and engagement with students and parents,  enhancing  school  administration  and  management,  and  ensuring  safety  and  emergency  communications  (Sadiku  et  al.,  2021).  However,  not  all  schools  can  achieve the best effect that effective information dissemination can provide. Schools  in the Philippines face several challenges with information dissemination due to their  diverse culture and unique geographical and educational landscape. Hence, it is  essential to use the current information infrastructure and technology, such as the  Internet of Things (IoT), to enhance information dissemination within schools in the  Philippines.   Schools utilize multiple applications and social networking sites to disseminate  information to their students, parents, and school members. Social media sites like  Facebook are the most popular and commonly used. Facebook is a social media  platform founded by Mark Zuckerberg in 2004. It enables users to create profiles, share  pictures, and connect with families and friends. It also allows users to create pages  and groups to get updated with news and content. Sites like Twitter have similar  features,  albeit  rarely  used  by  schools.  Some  applications  allow  their  users  to  exchange information, similar to forums. Among them, the most popular one that most ",
    "One issue that many people nowadays face is that the job or school where they wish to work, or study is situated a long distance away from their homes. They must go from one island, city, or province to another, and many are unfamiliar with the region. Others possess real estate holdings that they want to rent out as a secondary source of income but have no idea how to advertise them properly. ARent, an online and mobile growth and economic",
    "Education plays a crucial role in shaping individuals and societies. It is the key that unlocks doors to knowledge, opportunity, and personal growth. Education empowers individuals with the skills and knowledge needed to navigate through life's challenges and pursue their goals. Beyond acquiring subject-specific knowledge, education equips individuals with critical thinking abilities, problem-solving skills, and a broad perspective on the world. It enables people to understand diverse cultures, appreciate different viewpoints, and promotes tolerance and empathy. Moreover, education is not limited to formal institutions; it encompasses lifelong learning and continuous personal development. In essence, education is the foundation upon which individuals build fulfilling lives and contribute meaningfully to their communities.",
]
rules = {}
rules["Goal 1"] = ['poverty', 'poor', 'inequality',
                   'income', 'disparity', 'reduction']
rules["Goal 2"] = ['hunger', 'malnutrition',
                   'famine', 'food', 'waste', 'farming']
rules["Goal 3"] = ['health', 'covid', 'mental',
                   'vaccination', 'disease', 'pandemic', 'healthcare']
rules["Goal 4"] = ['education', 'literacy',
                   'quality', 'learning', 'school', 'development']
rules["Goal 5"] = ['gender', 'equality', 'empowerment',
                   'women', 'rights', 'equal', 'protection', 'lgbtq']
rules["Goal 6"] = ['water', 'clean', 'sanitation',
                   'conservation', 'hygience', 'scarcity']
rules["Goal 7"] = ['energy', 'renewable',
                   'fuel', 'fossil', 'reduction', 'clean']
rules["Goal 8"] = ['work', 'growth', 'economic',
                   'labor', 'rights', 'productivity', 'job']
rules["Goal 9"] = ['innovation', 'artificial', 'intelligence',
                   'AI', 'technology', 'infrastructure', 'industrialization']
rules["Goal 10"] = ['inequality', 'racism', 'racist',
                    'empowerment', 'marginalized', 'equity']
rules["Goal 11"] = ['cities', 'sustainable',
                    'urbanization', 'green', 'spaces', 'resilient']
rules["Goal 12"] = ['consumption', 'sustainable',
                    'resource', 'management', 'waste']
rules["Goal 13"] = ['climate', 'change', 'carbon', 'emissions', 'energy']
rules["Goal 14"] = ['marine', 'conservation', 'coral',
                    'reefs', 'biodiversity', 'ocean', 'pollution']
rules["Goal 15"] = ['land', 'life', 'animals', 'wildlife',
                    'degradation', 'ecosystem', 'deforestation', 'landslide', 'plant', 'specie']
rules["Goal 16"] = ['peace', 'justice',
                    'institution', 'corruption', 'law', 'rule', 'society']
rules["Goal 17"] = ['partnerships', 'collaboration',
                    'cooperation', 'capacity', 'goals', 'sustainable']


class ONLY:

    def main(self):
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
                data = self.getTFIDF(appendedData)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Final Execution time:", execution_time, "seconds")

    def getTFIDF(self, data):
        newDocs = []
        trainingDocs = cons.extractTraining()
        newDocs.append(data)
        newData = cons.preprocess_documents(newDocs)
        data = newData[0]
        trainingDocs.append(data)
        listOfDict = cons.TFIDFForConfusion(trainingDocs, False)
        count = 1
        newDoc = listOfDict[len(listOfDict)-1]
        print(newDoc)
        del listOfDict[-1]
        result = self.compare(newDoc)
        return result

    def compare(self, dic):
        total_goal, temp = [], []
        total = 1
        testing, final, super_final_dict = {}, {}, {}
        values = list(dic.keys())
        rules_values = list(rules.values())
        for rules1 in rules_values:
            for rules2 in rules1:
                for val in values:
                    if (rules2 == val):
                        testing[dic[val]] = total
                        temp.append(dic[val])
            total += 1
            total_goal.append(temp)

        for key, value in testing.items():
            str_value = str(value)
            if value != -0.0 and str_value != '-0.0':
                if value not in final:
                    final[value] = key
                else:
                    final[value] += key

        for test in final:
            if test == 1:
                super_final_dict['Goal 1: No Poverty'] = final[test]
            elif test == 2:
                super_final_dict['Goal 2: Zero Hunger'] = final[test]
            elif test == 3:
                super_final_dict['Goal 3: Good Health and Well-Being'] = final[test]
            elif test == 4:
                super_final_dict['Goal 4: Quality Education'] = final[test]
            elif test == 5:
                super_final_dict['Goal 5: Gender Equality'] = final[test]
            elif test == 6:
                super_final_dict['Goal 6: Clean Water and Sanitation'] = final[test]
            elif test == 7:
                super_final_dict['Goal 7: Affordable and Clean Energy'] = final[test]
            elif test == 8:
                super_final_dict['Goal 8: Decent Work and Economic Growth'] = final[test]
            elif test == 9:
                super_final_dict['Goal 9: Industry, Innovation, and Infrastructure'] = final[test]
            elif test == 10:
                super_final_dict['Goal 10: Reduced Inequalities'] = final[test]
            elif test == 11:
                super_final_dict['Goal 11: Sustainable Cities and Communities'] = final[test]
            elif test == 12:
                super_final_dict['Goal 12: Responsible Consumption and Production'] = final[test]
            elif test == 13:
                super_final_dict['Goal 13: Climate Action'] = final[test]
            elif test == 14:
                super_final_dict['Goal 14: Life Below Water'] = final[test]
            elif test == 15:
                super_final_dict['Goal 15: Life on Land'] = final[test]
            elif test == 16:
                super_final_dict['Goal 16: Peace, Justice and Strong Institutions'] = final[test]
            elif test == 17:
                super_final_dict['Goal 17: Partnership for the Goals'] = final[test]
        sorted_dict = dict(
            sorted(super_final_dict.items(), key=lambda item: item[1], reverse=True))

        return sorted_dict

        # total += 1
        # total_goal.append(temp)
        # temp.clear

        # for val in values:
        #     print(val)
        # for goal in rules.values():
        #     for str in dic:
