import numpy as np
import pandas as pd
import yaml
import os
import shutil
import tensorflow as tf
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


def train_and_test(config_file):
    config=get_data(config_file)
    rescale= config['img_augment']['rescale']
    shear_range= config['img_augment']['shear_range']
    zoom_range=config['img_augment']['zoom_range']
    horizontal_flip=config['img_augment']['horizontal_flip']
    vertical_flip= config['img_augment']['vertical_flip']
    batch_size=config['img_augment']['batch_size']
    class_mode= config['img_augment']['class_mode']
    train_path=config['model']['train_path']
    test_path=config['model']['test_path']
    image_size=config['model']['image_size']
    activationone=config['model']['activationone']
    activationtwo=config['model']['activationtwo']
    final_neuron=config['model']['final_neuron']
    loss=config['model']['loss']
    optimizer=config['model']['optimizer']
    epochs=config['model']['epochs']
    metrics=config['model']['metrics']
    model_path=config['model']['sav_dir']
    val_path=config['model']['val_path']


    model = Sequential([
    Conv2D(32, (3, 3), activation=activationone, input_shape=(255, 255, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation=activationone),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation=activationone),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation=activationone),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation=activationone),
    Dropout(0.5),
    Dense(final_neuron, activation=activationtwo)  
])
    datagen=ImageDataGenerator(rescale=rescale
                               ,shear_range=shear_range,
                               zoom_range=zoom_range,
                               horizontal_flip=horizontal_flip
                               ,vertical_flip=vertical_flip)
    train_data=datagen.flow_from_directory(train_path,batch_size=batch_size,class_mode=class_mode,target_size=image_size)
    val_gen=ImageDataGenerator(rescale=rescale)
    val_data=val_gen.flow_from_directory(val_path,batch_size=batch_size,class_mode=class_mode,target_size=image_size)

    
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
   
    model.fit(train_data,epochs=epochs,validation_data=val_data)
    
    model.save(model_path)
    


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config_file=passed_args.config)