import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import shutil
import os
import argparse
import tensorflow as tf
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from get_data import get_data, read_params
from tensorflow.keras. preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


def evaluate(config_file):
    config=get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']
    model = load_model('models/trained.h5')
    config = get_data(config_file)

    test_gen = ImageDataGenerator(rescale = 1./224)
    test_set = test_gen.flow_from_directory(te_set,
                                                target_size = (224,224),
                                                batch_size = batch,
                                                class_mode = class_mode
                                                )
    
    Y_pred = model.predict(test_set, len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)
    print("Confusion Matrix")
    sns.heatmap(confusion_matrix(test_set.classes,y_pred),annot=True)
    plt.xlabel('Actual vlaues,0:Bulbasaur, 1:Charmander, 2: Squirtle,3:Tauros')
    plt.ylabel('Predicted Value,0:Bulbasaur, 1:Charmander, 2: Squirtle,3:Tauros')
    plt.savefig('reports/Confusion_Matrix')
    #plt.show()

    print("Classification Report")
    target_names = ['Accident','Non Accident']
    df =pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)).T
    df['support']=df.support.apply(int)
    df.style.background_gradient(cmap='viridis',subset=pd.IndexSlice['0':'9','f1-score'])
    df.to_csv('reports/classification_report')
    print('Classification Report and Confusion Matrix Report are saved in reports folder of Template')


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    evaluate(config_file=passed_args.config)



