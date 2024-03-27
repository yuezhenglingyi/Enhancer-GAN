import random, os, h5py, math, time, glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from utils.utils import *
from utils.bio_utils import *
from utils.lang_utils import *

### Load libraries

from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History

import pandas as pd
import numpy as np

import sys
from utils.Neural_Network_DNA_Demo.helper import  IOHelper, SequenceHelper # from https://github.com/bernardo-de-almeida/Neural_Network_DNA_Demo.git

### load model
def load_model(model_path):
    import deeplift
    from keras.models import model_from_json
    keras_model_weights = model_path + '.h5'
    keras_model_json = model_path + '.json'
    # success
    keras_model = model_from_json(open(keras_model_json).read())
    keras_model.load_weights(keras_model_weights)
    #keras_model.summary()
    return keras_model, keras_model_weights, keras_model_json

def Deep_STARR_pred_new_sequence(new_seq, model_ID):
    
    if new_seq=='': sys.exit("fasta seq file not found")
    if model_ID=='': sys.exit("CNN model file not found")
    print('Input FASTA file is ', new_seq)
    # print('Model file is ', model_ID)

    sys.path.append('Neural_Network_DNA_Demo/')

    ### Load sequences
    print("\nLoading sequences ...\n")
    input_fasta = IOHelper.get_fastas_from_file(new_seq, uppercase=True)
    print(input_fasta.shape)
    # print(len(input_fasta[0][1]))

    # length of first sequence
    sequence_length = len(input_fasta.sequence.iloc[0])
    # print(sequence_length)
    # Convert sequence to one hot encoding matrix
    seq_matrix = SequenceHelper.do_one_hot_encoding(input_fasta.sequence, sequence_length,
                                                    SequenceHelper.parse_alpha_to_seq)
    # print(seq_matrix.shape)
    keras_model, keras_model_weights, keras_model_json = load_model(model_ID)

    ### predict dev and hk activity
    print("\nPredicting ...")
    pred=keras_model.predict(seq_matrix)
    out_prediction = input_fasta
    out_prediction['Predictions_dev'] = pred[0]
    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # e = 0
    # f = 0
    # g = 0
    # h = 0
    # j = 0
    # k = 0
    # for i in range(len(pred[0])):
    #     if pred[0][i] < -3:
    #         f = f + 1
    #     elif -3 < pred[0][i] < -2:
    #         g = g + 1
    #     elif -2 < pred[0][i] < -1:
    #         h = h + 1
    #     elif -1 < pred[0][i] < 0:
    #         a = a + 1
    #     elif 0 < pred[0][i] < 1:
    #         b = b + 1
    #     elif 1 < pred[0][i] < 2:
    #         c = c + 1
    #     elif 2 < pred[0][i] < 3:
    #         d = d + 1
    #     elif pred[0][i] > 5:
    #         e = e + 1
    # print("<-3:", f, " -3~-2:", g, " -2~-1", h, ' -1~0:',a ,' 0~1:',b ,' 1~2:',c ,' 2~3:',d ,' >3:',e)
    out_prediction['Predictions_hk'] = pred[1]
    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # e = 0
    # f = 0
    # g = 0
    # h = 0
    # j = 0
    # k = 0
    # for i in range(len(pred[1])):
    #     if pred[1][i] < -3:
    #         f = f + 1
    #     elif -3 < pred[1][i] < -2:
    #         g = g + 1
    #     elif -2 < pred[1][i] < -1:
    #         h = h + 1
    #     elif -1 < pred[1][i] < 0:
    #         a = a + 1
    #     elif 0 < pred[1][i] < 1:
    #         b = b + 1
    #     elif 1 < pred[1][i] < 2:
    #         c = c + 1
    #     elif 2 < pred[1][i] < 3:
    #         d = d + 1
    #     elif pred[1][i] > 5:
    #         e = e + 1
    # print("<-3:", f, " -3~-2:", g, " -2~-1", h, ' -1~0:',a ,' 0~1:',b ,' 1~2:',c ,' 2~3:',d ,' >3:',e)

    ### save file
    print("Saving file ...\n")
    import os.path
    model_ID_out=os.path.basename(model_ID)
    out_prediction.to_csv(new_seq + "_predictions_" + model_ID_out + ".txt", sep="\t", index=False)


def main():
    sample = './samples/' + 'realProt_50aa' + "/sampled_53999.txt"
    with open(sample, 'r') as f1, \
        open("sample_out.txt", 'w+') as f2:
        f2.writelines(['>no_location' +'\n' + line +  '\n' for line in f1])
    Deep_STARR_pred_new_sequence("sample_out.txt", "DeepSTARR.model")

if __name__ == '__main__':
    main()
