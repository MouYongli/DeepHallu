from curses import raw
import os
import os.path as osp
import json
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

HERE = osp.dirname(osp.abspath(__file__))
DATA_ROOT_PATH = osp.join(HERE, '..', '..', '..', 'data', 'mme')
DATA_PATH = osp.join(DATA_ROOT_PATH, 'MME_Benchmark_release_version', 'MME_Benchmark')
CATEGORIES = ['artwork', 'celebrity', 'code_reasoning', 'color', 'commonsense_reasoning', 'count', 'existence', 'landmark', 'numerical_calculation', 'OCR', 'position', 'posters', 'scene', 'text_translation']

class MMEDataset(Dataset):
    def __init__(self, data_path: str = None, categories: list = None):
        if data_path is None:
            self.data_path = DATA_PATH
        else:
            self.data_path = data_path
        if categories is None:
            self.categories = CATEGORIES
        else:
            self.categories = categories
        preprocessed_data = self.load_preprocessed_data(self.categories)
        self.data = pd.DataFrame(preprocessed_data)
    
    def load_preprocessed_data(self, categories: list = None):
        """
        Load the QA list from the data path.
        """
        preprocessed_data = []
        if osp.exists(osp.join(self.data_path, 'preprocessed.json')):
            with open(osp.join(self.data_path, 'preprocessed.json'), 'r') as f:
                preprocessed_data = json.load(f)
        else:
            raise FileNotFoundError(f"Preprocessed QA data not found at {self.data_path}")
        return preprocessed_data

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = osp.join(self.data_path, item['image_path'])
        image = Image.open(image_path)
        return image, item['id'], item['image_name'], item['category'], item['question'], item['answer']
    
    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    dataset = MMEDataset()
    print(dataset[0])