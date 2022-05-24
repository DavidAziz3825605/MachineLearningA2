# Imports

import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from data_loader import DataLoader
import pickle
import sys
import matplotlib.pyplot as plt


MODELS_SAVE_PATH = 'models/RF'
INDEPENDENT_DATA_PATH = "dataset/trafficsigns_dataset_independent_preprocessed_gray"



class RF:
    def __init__(self) -> None:
        self.dataset = DataLoader()


    def train(self, signTypes = False):
        # use K-Folds cross-validator 
        KF = KFold(n_splits=25, shuffle=True)
        train_results = [] 
        X = np.array(self.dataset.images_data.drop\
                                    (self.dataset.images_data.columns[[0,1]], axis = 1))
        # store y labels for both sign type and sign shape classes
        if signTypes:
            y = np.array(self.dataset.images_data[self.dataset.images_data.columns[1]])
        else:
            y = np.array(self.dataset.images_data[self.dataset.images_data.columns[0]])

        # train the model
        print('_'*150)
        if signTypes:
           print( "Training RF classifier on sign type classes\n")
        else:
            print("Training RF classifier on sign shape classes\n")
        print("\nStarted Training")
        model = RandomForestClassifier(n_estimators = 100, verbose=1)        
        for train_index, test_index in KF.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_results.append(f1_score(y_test, y_pred, average = 'weighted'))
        
        print("Training results\n")
        print("F1-score: ", np.mean(train_results))

        print("Saving model for later inference......")
        if signTypes:
            model_file = "RF_SignTypes.pickle"
        else:
            model_file = "RF_SignShapes.pickle"
        
        if not os.path.isdir(MODELS_SAVE_PATH):
            os.makedirs(MODELS_SAVE_PATH)
        with open(os.path.join(MODELS_SAVE_PATH, model_file), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved at {os.path.join(MODELS_SAVE_PATH, model_file)}.")
        print('_'*150)

    def independent_eval(self):
        # load sign shape classifier
        with open(os.path.join(MODELS_SAVE_PATH, 'RF_SignShapes.pickle'), 'rb') as handle:
            shape_model = pickle.load(handle)

        # load sign type classifier
        with open(os.path.join(MODELS_SAVE_PATH, 'RF_SignTypes.pickle'), 'rb') as handle:
            type_model = pickle.load(handle)

        X_indepent_eval_data = np.array(self.dataset.independent_data.drop\
                                    (self.dataset.independent_data.columns[[0,1]], axis = 1))

        # load y labels for both categories
        y_indepent_eval__shape =\
             np.array(self.dataset.independent_data[self.dataset.independent_data.columns[0]])

        y_indepent_eval__type =\
             np.array(self.dataset.independent_data[self.dataset.independent_data.columns[1]])

        # eval on sign shape independent data
        y_pred_shape = shape_model.predict(X_indepent_eval_data)
        shape_pred_score = f1_score(y_indepent_eval__shape, y_pred_shape, average = 'weighted')
        shape_pred_score = round(shape_pred_score,2)
        # eval on sign type independent data
        y_pred_type = type_model.predict(X_indepent_eval_data)
        type_pred_score = f1_score(y_indepent_eval__type, y_pred_type, average = 'weighted')

        type_pred_score = round(type_pred_score, 2)
        
        print(f"F1-score for independent evaluation on sign shape classes: {shape_pred_score}")
        print('\n')
        print(f"F1-score for independent evaluation on sign type classes: {type_pred_score}")
        self.plot_save_results(X_indepent_eval_data,
                                y_pred_shape, y_pred_type,
                                y_indepent_eval__shape,
                                y_indepent_eval__type,
                                shape_pred_score,
                                type_pred_score)


    def plot_save_results(self,images, y_pred_shape, y_pred_type,
                            y_gt_shape, y_gt_type, shape_f1, type_f1):

        # create folder to save plots
        save_path = "Results/RF"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # create figure for sign shape results
        images = images.reshape((images.shape[0], 28, 28))
        plt.figure(figsize=(100, 100))

        for i, img in enumerate(images):
            plt.subplot(18, 5, i+1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            pred_shape = self.dataset.shape_classes_labels[y_pred_shape[i]]
            shape_gt = self.dataset.shape_classes_labels[y_gt_shape[i]]

            pred_type = self.dataset.type_classes_labels[y_pred_type[i]]
            type_gt = self.dataset.type_classes_labels[y_gt_type[i]]

            img_title_x = f"predicted shape: {pred_shape}, shape GT: {shape_gt}"
            
            # img_title_y = f"predicted type: {pred_type}, type GT: {type_gt}"
            if (pred_shape==shape_gt):
                plt.xlabel(img_title_x, color='g', fontsize=40)
            else:
                plt.xlabel(img_title_x, color='r', fontsize=40)
            plt.imshow(img, cmap='gray')
        plt.suptitle(f'sign shape independent eval, F1-Score: {shape_f1}',fontsize=100)
        plt.savefig(os.path.join(save_path, 'sign shape independent eval results.pdf'))

        for i, img in enumerate(images):
            plt.subplot(18, 5, i+1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            
            pred_type = self.dataset.type_classes_labels[y_pred_type[i]]
            type_gt = self.dataset.type_classes_labels[y_gt_type[i]]

            
            img_title_x = f"predicted type: {pred_type}, type GT: {type_gt}"
            if (pred_type==type_gt):
                plt.xlabel(img_title_x, color='g', fontsize=40)
            else:
                plt.xlabel(img_title_x, color='r', fontsize=40)
            plt.imshow(img, cmap='gray')
        plt.suptitle(f'sign type independent eval, F1-Score: {type_f1}',fontsize=100)
        plt.savefig(os.path.join(save_path, 'sign type independent eval results.pdf'))




if __name__ == "__main__":

    
    RF_classifier = RF()
    try:
        opt = arguments = sys.argv[1]
        if opt == "train":
            RF_classifier.train()
            RF_classifier.train(signTypes=True)
        elif opt == "eval":
            RF_classifier.independent_eval()
    except:
        print("You need to pass system argument (train || eval)")