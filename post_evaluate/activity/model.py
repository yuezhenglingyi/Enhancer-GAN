import tensorflow as tf

import keras
import numpy as np
import keras.layers as kl
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )

params = {'batch_size': 128,
          'epochs': 100,
          'early_stop': 10,
          'kernel_size1': 7,
          'kernel_size2': 3,
          'kernel_size3': 5,
          'kernel_size4': 3,
          'lr': 0.002,
          'num_filters': 256,
          'num_filters2': 60,
          'num_filters3': 60,
          'num_filters4': 120,
          'n_conv_layer': 4,
          'n_add_layer': 2,
          'dropout_prob': 0.4,
          'dense_neurons1': 256,
          'dense_neurons2': 256,
          'pad':'same'}

def DeepSTARR(params=params):
    
    lr = params['lr']
    dropout_prob = params['dropout_prob']
    n_conv_layer = params['n_conv_layer']
    n_add_layer = params['n_add_layer']
    
    # body
    input = kl.Input(shape=(249, 4))
    x = kl.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'], padding=params['pad'], name='Conv1D_1st')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    for i in range(1, n_conv_layer):
        x = kl.Conv1D(params['num_filters'+str(i+1)], kernel_size=params['kernel_size'+str(i+1)], padding=params['pad'], name=str('Conv1D_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
    
    x = Flatten()(x)
    embeddings = x
    # dense layers
    for i in range(0, n_add_layer):
        x = kl.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_prob)(x)
    bottleneck = x
    
    # heads per task (developmental and housekeeping enhancer activities)
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))
    # print(len(outputs))
    # print(type(outputs[0]))
    # print(keras.backend.shape(outputs[0]))
    # print(outputs[0].shape[0])
    # print(outputs[0].shape[1])
    model = keras.models.Model(inputs=[input], outputs=[outputs, embeddings])

    model.compile(keras.optimizers.adam_v2.Adam(lr=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman]) # additional track metric

    return model, params


def tSNE(real_samples, fake_samples, name):

    # DeepSTARR for embedding
    model, model_params = DeepSTARR()
    pre_trained_model_path = "./Assess/pre_trained_model/DeepSTARR.model.h5"
    model.load_weights(pre_trained_model_path)

    real_pre, real_embedding = model.predict(real_samples, batch_size=128)
    fake_pre, fake_embedding = model.predict(fake_samples, batch_size=128)

    # t-SNE analysis on real_sample and fake_sample
    real_sample_number = len(real_embedding)
    fake_sample_number = len(fake_embedding)

    RS = 42
    sample_embedding = np.concatenate([real_embedding, fake_embedding], axis=0)
    sample_embedding_proj = TSNE(n_components=2, init='pca', random_state=RS).fit_transform(sample_embedding)
    
    real_embedding_proj = sample_embedding_proj[0: real_sample_number]
    fake_embedding_proj = sample_embedding_proj[real_sample_number: real_sample_number+fake_sample_number]

    real_marker = "o"
    fake_marker = "*"

    real_colors = ["#48CAE4", "#3A86FF"]
    fake_colors = ["#CCFF33", "#006400"]

    real_c = []
    fake_c = []
    for activity in real_pre[0]:
        if activity > 2.0:
            real_c.append(1)
        else:
            real_c.append(0)

    for activity in fake_pre[0]:
        if activity > 2.0:
            fake_c.append(1)
        else:
            fake_c.append(0)

    # t-SNE image
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(real_embedding_proj[:, 0], real_embedding_proj[:, 1], marker=real_marker, c=np.array(real_colors)[np.array(real_c)], alpha=0.2, s=20, label="real_sample")
    ax.scatter(fake_embedding_proj[:, 0], fake_embedding_proj[:, 1], marker=fake_marker, c=np.array(fake_colors)[np.array(fake_c)], alpha=0.2, s=20, label="fake_sample")
        
    plt.xticks([])
    plt.yticks([])
    plt.title("name".format(name))
    plt.rcParams.update({'font.size': 15})
    plt.legend(loc="upper left")
    plt.savefig("image/{}.svg".format(name), bbox_inches='tight')
    plt.show()
    # plt.close()