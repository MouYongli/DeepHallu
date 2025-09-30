import os
import os.path as osp
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = osp.dirname(osp.abspath(__file__))
DATA_ROOT_PATH = osp.join(HERE, '..', '..', '..', 'data', 'mme')
DATA_PATH = osp.join(DATA_ROOT_PATH, 'MME_Benchmark_release_version', 'MME_Benchmark')
CATEGORIES = ['artwork', 'celebrity', 'code_reasoning', 'color', 'commonsense_reasoning', 'count', 'existence', 'landmark', 'numerical_calculation', 'OCR', 'position', 'posters', 'scene', 'text_translation']

"""
MME Benchmark is a benchmark for multimodal large language models. This script is used to preprocess the data for the MME Benchmark.
Save the QA list to the data path, which includes the following information:
- id: int, the id of the question
- category: str, the category of the question
- image_name: str, the name of the image
- image_format: str, the format of the image
- image_path: str, the path to the image
- question: str, the question
- answer: str, the answer
"""

class MMEPreprocessor:
    """
    MME Preprocessor.
    MME Benchmark is a benchmark for multimodal large language models.
    Args:
        data_path: str, the path to the data
        image_path: str, the path to the images
        categories: list, the categories of the data
    """
    def __init__(self, data_path: str = None, categories: list = None):
        if data_path is None:
            self.data_path = DATA_PATH
        else:
            self.data_path = data_path
        if categories is None:
            self.categories = CATEGORIES
        else:
            self.categories = categories
        self.qa_list = self.load_qa_list()

    def load_qa_list(self):
        """
        Load the QA list from the data path.
        """
        qa_list = []
        if osp.exists(osp.join(self.data_path, 'qa.json')):
            with open(osp.join(self.data_path, 'qa.json'), 'r') as f:
                qa_list = json.load(f)
        else:
            idx = 0
            for category in self.categories:
                if osp.exists(osp.join(self.data_path, category, 'images')) and osp.exists(osp.join(self.data_path, category, 'questions_answers_YN')):
                    images = [f for f in os.listdir(osp.join(self.data_path, category, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
                    for image in images:
                        image_name = image.split('.')[0]
                        image_format = image.split('.')[-1]
                        image_path = osp.join(category, 'images', image)
                        qa_path = osp.join(self.data_path, category, 'questions_answers_YN', f'{image_name}.txt')
                        with open(qa_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                question = line.split('\t')[0].strip()
                                answer = line.split('\t')[1].strip()
                                qa_list.append({
                                    'id': idx,
                                    'category': category,
                                    'image_name': image_name,
                                    'image_format': image_format,
                                    'image_path': image_path,
                                    'question': question,
                                    'answer': answer
                                })
                                idx += 1
                else:
                    images = [f for f in os.listdir(osp.join(self.data_path, category)) if f.endswith('.jpg') or f.endswith('.png')]
                    for image in images:
                        image_name = image.split('.')[0]
                        image_format = image.split('.')[-1]
                        image_path = osp.join(category, image)
                        qa_path = osp.join(self.data_path, category, f'{image_name}.txt')
                        with open(qa_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                question = line.split('\t')[0].strip()
                                answer = line.split('\t')[1].strip()
                                qa_list.append({
                                    'id': idx,
                                    'category': category,
                                    'image_name': image_name,
                                    'image_format': image_format,
                                    'image_path': image_path,
                                    'question': question,
                                    'answer': answer
                                })
                                idx += 1
        return qa_list

    def save_qa_list(self):
        """
        Save the QA list to the data path.
        """
        with open(osp.join(self.data_path, 'qa.json'), 'w') as f:
            json.dump(self.qa_list, f, indent=2, ensure_ascii=False)

    def preprocess_data(self):
        pass

if __name__ == "__main__":
    preprocessor = MMEPreprocessor()
    preprocessor.save_qa_list()