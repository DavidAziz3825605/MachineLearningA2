
import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import os

DATA_CSV_SAVE_PATH = "dataset/trafficsigns_dataset_csv"
INDPENDENT_DATA_CSV = "dataset/trafficsigns_dataset_independent_csv"

class DataLoader:
    """
        Loads the training data and independent evaluation data in single object
    """
    def __init__(self) -> None:
        self.DataPath = DATA_CSV_SAVE_PATH
        self.IndependentDataPath = INDPENDENT_DATA_CSV
        self.shape_classes_labels = self.read_labelMap("IDs_shapes.csv")
        self.type_classes_labels = self.read_labelMap("IDs_types.csv")
        self.images_data = self.load_images_df()
        self.independent_data = self.load_independent_images_df()
    
    def read_labelMap(self, file):
        """reads the dataframe whcih contains classes name alongside numerical labels

        Args:
            file (str): sign shape or sign tye

        Returns:
            dict : dataframe columns refactored inti dict
        """
        df = pd.read_pickle(os.path.join(self.DataPath, file))
        classes = list(df['class'])
        ids = list(df['ID'])
        label_map = dict()
        for i, id in enumerate(ids):
            label_map[id] = classes[i]
        return label_map


    def load_images_df(self):
        images_df = pd.read_pickle(os.path.join(self.DataPath, "images_data.csv"))
        return images_df

    def load_independent_images_df(self):
        images_df = pd.read_pickle(os.path.join(self.IndependentDataPath, "images_data.csv"))
        return images_df
