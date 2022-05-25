# Imports
import os
from unittest.main import main
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd

os.sys.path
os.chdir('./')

# paths to load ans save data

# original data-set
DATA_PATH = 'dataset/trafficsigns_dataset'
# dataset after converted to csv format
DATA_CSV_SAVE_PATH= "dataset/trafficsigns_dataset_csv"
# original independent data-set
INDPENDENT_DATA_PATH = "dataset/trafficsigns_dataset_independent"
# independent data-set after preprocessing
INDPENDENT_DATA_PREPROCESSED = "dataset/trafficsigns_dataset_independent_preprocessed"
# preprocessed independent data-set converted to csv format
INDPENDENT_DATA_CSV = "dataset/trafficsigns_dataset_independent_csv"

def read_traffic_data_dict(path):
    """This functions loads the dataset
        and categorise it in dictionary of numpy array

    Args:
        path (sring): dataset main folder path
    """
    # define dataset dictionary
    data_dict = dict()
    for c in os.listdir(path):
        # check directories only
        if not os.path.isdir(os.path.join(path, c)):
            continue
        # assign empty dictionary at first for each shape class to store type classes
        data_dict[c] = {'signTypes': dict()}
        # join paths to get each sign shape class path
        signShapePath = os.path.join(path, c)
        # loop over sign types classes in each sign shape class
        for k in os.listdir(signShapePath):
            # join paths to get each sign type class path
            signTypePath = os.path.join(signShapePath, k)
            # check directories only
            if os.path.isdir(signTypePath):
                signTypeArr = list()
                for img_file in os.listdir(signTypePath):
                    
                    img = cv2.imread(os.path.join(signTypePath, img_file), 0)
                    try:
                        img_flattened = img.flatten()
                    except:
                        continue
                    signTypeArr.append(img_flattened)
                data_dict[c]['signTypes'][k] = signTypeArr
    return data_dict



def create_labels_map(data_path, save_path):
    """ creates a numerical value for each categorical class ti be used in model evaluation
        the result is a pandas dataframe contains 2 cols, one for name and the other for numerical id 
    Args:
        data_path (str): dataset path
        save_path (str): path to save resulted dataframes

    Returns:
        resulted dataframes: resulted dataframes
    """
    # list to store sign shape classes names
    sign_shapes = []
    # list to store sign type classes names
    sign_types = [] 
    for dir in os.listdir(data_path):
        # check directories only
        sign_shape_class_path = os.path.join(data_path, dir)
        if not os.path.isdir(sign_shape_class_path):
            continue

        sign_shapes.append(dir)
        for sub_dir in os.listdir(sign_shape_class_path):
            sign_type_class_path = os.path.join(sign_shape_class_path, sub_dir)
            if not os.path.isdir(sign_type_class_path):
                continue
            sign_types.append(sub_dir)
    # make pandas dataframe of sign type and sign shape classes with numerical labels
    shape_classes_map = pd.DataFrame(
        {
            'class': sign_shapes,
            'ID': list(range(0, len(sign_shapes)))
        })

    type_classes_map = pd.DataFrame(
        {
            'class': sign_types,
            'ID': list(range(0, len(sign_types)))
        })
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # save created dataframes
    shape_classes_map.to_pickle(os.path.join(save_path, f"IDs_shapes.csv"))
    type_classes_map.to_pickle(os.path.join(save_path, f"IDs_types.csv"))
    return shape_classes_map, type_classes_map



def save_traffic_data_csv(data_path, save_csv_data_path):
    """ created a pandas dataframe for all images and labels in single dataframe
        col 0 : sign shape label
        col 1 : sign type label
        from col 2 to col 786 : images pixel gray values

    Args:
        data_path (str): dataset path
        save_csv_data_path (str): path to save resulted dataframes
    """
    shape_classes_map, type_classes_map = create_labels_map(data_path, save_csv_data_path)
    data_dict = read_traffic_data_dict(data_path)

    all_images = list()
    # loop over all shape classes
    for shape_class, shapes in data_dict.items():
        
        # the label of the shape
        shape_id = int(shape_classes_map[shape_classes_map['class'] == shape_class]['ID'].values[0])

        # create labels for types in particular shape class


        for type_class, images in shapes['signTypes'].items():

            type_id = int(type_classes_map[type_classes_map['class'] == type_class]['ID'].values[0])
           
            for img in images:
                img_with_labels = img/255
                img_with_labels = np.insert(img_with_labels, 0, type_id)
                img_with_labels = np.insert(img_with_labels, 0, shape_id)
                all_images.append(img_with_labels)
    
    images_df = pd.DataFrame(all_images)
    images_df.to_pickle(os.path.join(save_csv_data_path, "images_data.csv"))


def preprocess_independednt_data(data_path, save_path):
    """resize indepenedent data images to be 28*28 and convert to gray scale

    Args:
        data_path (str): data folder images
        path (str): saved images directory
        
    """
    for c in os.listdir(data_path):
        # check directories only
        if not os.path.isdir(os.path.join(data_path, c)):
            continue
        # join paths to get each sign shape class path
        signShapePath = os.path.join(data_path, c)
        signShapeSavePath = os.path.join(save_path, c)

        for k in os.listdir(signShapePath):
            # join paths to get each sign type class path
            signTypePath = os.path.join(signShapePath, k)
            signTypeSavePath = os.path.join(signShapeSavePath, k)
            # check directories only
            if os.path.isdir(signTypePath):
                if not os.path.isdir(signTypeSavePath):
                    os.makedirs(signTypeSavePath)
                for img_file in os.listdir(signTypePath):
                    img = cv2.imread(os.path.join(signTypePath, img_file), 0)
                    try:
                        result_image = cv2.resize(img, (28,28))
                    except:
                        continue
                    cv2.imwrite(os.path.join(signTypeSavePath, img_file), result_image)



if __name__ == "__main__":
    save_traffic_data_csv(DATA_PATH, DATA_CSV_SAVE_PATH)
    preprocess_independednt_data(INDPENDENT_DATA_PATH, INDPENDENT_DATA_PREPROCESSED)
    save_traffic_data_csv(INDPENDENT_DATA_PREPROCESSED, INDPENDENT_DATA_CSV)