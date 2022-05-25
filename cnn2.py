import numpy as np
import random
import os
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
import keras.backend as K
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import f1_score
import sys
from data_loader import DataLoader

MODELS_SAVE_PATH = 'models/CNN2'

class modified_VGG16:
    def __init__(self) -> None:
        self.dataset = DataLoader()
        self.nof_shape_classes = len(self.dataset.shape_classes_labels)
        self.nof_type_classes = len(self.dataset.type_classes_labels)
        self.input_dim = 28
        self.shape_model = self.build_model(self.nof_shape_classes)
        self.type_model = self.build_model(self.nof_type_classes)
        self.epochs = 50


    def F1_Score(self, y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val


    def build_model(self, nof_outputs, print_summary=True):
        # Building the model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(self.input_dim,self.input_dim,1)),
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(nof_outputs, activation = 'softmax')
            ])

        # Compilation of the model
        model.compile(optimizer='SGD',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[self.F1_Score])

        if print_summary:
            # Model architecture
            model.summary() 
        return model



    def save_model_plots(self, history, save_path, model_type):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # save model evaluation plots
        plt.figure()
        plt.plot(history.history['F1_Score'], label='F1-Score')
        plt.plot(history.history['val_F1_Score'], label = 'Val F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.ylim([0.8, 1])
        plt.legend(loc='lower right')
        plt.title(f'CNN sign {model_type} classifier model F1-Score')
        plt.savefig(os.path.join(save_path, f'CNN sign {model_type}  classifier model F1-Score.png'))
        
        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 0.2])
        plt.legend(loc='lower right')
        plt.title(f'CNN sign {model_type} classifier model loss')
        plt.savefig(os.path.join(save_path, f'CNN sign {model_type} classifier model loss.png'))
    
    def train_shape_classifier(self):
        print("Training sign shape classifier cnn model")
        X = np.array(self.dataset.images_data.drop\
                                    (self.dataset.images_data.columns[[0,1]], axis = 1))
        
        X = X.reshape((X.shape[0], 28, 28))
        y = np.array(self.dataset.images_data[self.dataset.images_data.columns[0]])
        # convert from numerical to categorical
        y = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # print('X_train shape:', X_train.shape)
        # print('y_train shape:', y_train.shape)
        # print('X_test shape:', X_test.shape)
        # print('y_test shape:', y_test.shape)

        # fit the model
        history = self.shape_model.fit(X_train, y_train, epochs=self.epochs,
                    validation_data=(X_test, y_test))

        val_loss, val_acc = self.shape_model.evaluate(X_test, y_test, verbose=2)
        print('\nValifdation accuracy:', val_acc)
        print('\nValidation loss:', val_loss)
        
        if not os.path.isdir(MODELS_SAVE_PATH):
            os.makedirs(MODELS_SAVE_PATH)
        
        self.shape_model.save(os.path.join(MODELS_SAVE_PATH,"shape_model"))
        self.save_model_plots(history, "Results/CNN2/ACC_Loss_Graphs", "shape")


    

    def train_type_classifier(self):
        print("Training sign type classifier cnn model")

        X = np.array(self.dataset.images_data.drop\
                                    (self.dataset.images_data.columns[[0,1]], axis = 1))
        
        X = X.reshape((X.shape[0], 28, 28))
        y = np.array(self.dataset.images_data[self.dataset.images_data.columns[1]])
        y = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # print('X_train shape:', X_train.shape)
        # print('y_train shape:', y_train.shape)
        # print('X_test shape:', X_test.shape)
        # print('y_test shape:', y_test.shape)

        history = self.type_model.fit(X_train, y_train, epochs=self.epochs,
                    validation_data=(X_test, y_test))

        val_loss, val_acc = self.type_model.evaluate(X_test, y_test, verbose=2)
        print('\nValifdation accuracy:', val_acc)
        print('\nValidation loss:', val_loss)
        if not os.path.isdir(MODELS_SAVE_PATH):
            os.makedirs(MODELS_SAVE_PATH)
        
        self.type_model.save(os.path.join(MODELS_SAVE_PATH,"type_model"))
        self.save_model_plots(history, "Results/CNN2/ACC_Loss_Graphs", "type")



    def independent_eval(self):
        # load sign shape and sign type classifier models
        shape_model = tf.keras.models.load_model(os.path.join(MODELS_SAVE_PATH,"shape_model"), custom_objects={'F1_Score': self.F1_Score})
        type_model = tf.keras.models.load_model(os.path.join(MODELS_SAVE_PATH,"type_model"), custom_objects={'F1_Score': self.F1_Score})

        X_indepent_eval_data = np.array(self.dataset.independent_data.drop\
                                    (self.dataset.independent_data.columns[[0,1]], axis = 1))
        X_indepent_eval_data = X_indepent_eval_data.reshape((X_indepent_eval_data.shape[0], 28, 28))

        # load y labels for both categories
        y_indepent_eval__shape =\
             np.array(self.dataset.independent_data[self.dataset.independent_data.columns[0]])
        y_indepent_eval__shape = to_categorical(y_indepent_eval__shape)

        y_indepent_eval__type =\
             np.array(self.dataset.independent_data[self.dataset.independent_data.columns[1]])
        y_indepent_eval__type = to_categorical(y_indepent_eval__type)

        # eval on sign shape independent data
        y_pred_shape = shape_model.predict(X_indepent_eval_data)
        # eval on sign type independent data
        y_pred_type = type_model.predict(X_indepent_eval_data)

        y_indepent_eval__shape = np.argmax(y_indepent_eval__shape, axis=-1)
        y_indepent_eval__type = np.argmax(y_indepent_eval__type, axis=-1)

        y_pred_shape = np.argmax(y_pred_shape, axis=-1)
        y_pred_type = np.argmax(y_pred_type, axis=-1)


        type_pred_score = f1_score(y_indepent_eval__type, y_pred_type, average = 'weighted')
        type_pred_score = round(type_pred_score, 2)
        
        shape_pred_score = f1_score(y_indepent_eval__shape, y_pred_shape, average = 'weighted')
        shape_pred_score = round(shape_pred_score,2)

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
        save_path = "Results/CNN2"
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

    
    CNN_classifier = modified_VGG16()
    try:
        opt = arguments = sys.argv[1]
        if opt == "train":
            CNN_classifier.train_shape_classifier()
            CNN_classifier.train_type_classifier()
        elif opt == "eval":
            CNN_classifier.independent_eval()
    except:
        print("You need to pass system argument (train || eval)")