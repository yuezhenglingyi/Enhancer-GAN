import os
import math
import numpy as np

def seq_to_one_hot(sequences, charmap):
    """
        Function: 
            convert nucleotide into one-hot coding; 
    """

    # print(np.array(sequences).shape)
    # for seq in sequences:
    #     if len(seq) != 249:
    #         print("error")
    #         print(len(seq))
    #         print(seq)
    temp = [[charmap[c] for c in seq.replace("\n", "")] for seq in sequences]
    temp = np.array(temp)
    # temp = np.concatenate([np.concatenate([np.array(charmap[c]) for c in seq]) for seq in sequences] )
    # temp = np.array(temp).reshape(5000, 249, 4)
    # print(temp)
    # sequences_onehot = np.array(temp, dtype="int32") # if "P" not in seq

    return temp

def Activity_predict(sequences):
    """
        Function:
            predict the activity of input sequences about Dev and HK
        
        Parameter:
            sequences; consists of nucleotides, A, C, G, T
        
        Result:
            activity, including Dev and HK. activity[0] is the list of Dev, and activity[1] is the list of HK.
    """
    from post_evaluate.activity.model import DeepSTARR

    charmap = dict()
    charmap["A"] = [1, 0, 0, 0]
    charmap["C"] = [0, 1, 0, 0]
    charmap["G"] = [0, 0, 1, 0]
    charmap["T"] = [0, 0, 0, 1]

    eval_model, eval_params = DeepSTARR()
    pre_train_model = "./post_evaluate/activity/pre_trained_model/DeepSTARR.model.h5"
    eval_model.load_weights(pre_train_model)
    sequences_onehot = seq_to_one_hot(sequences, charmap)
    pre = eval_model.predict(sequences_onehot, batch_size=eval_params['batch_size'])   # [0]: Dev; [1]: HK
    
    sequences_activity = [pre[0][0], pre[0][1]]
    
    return sequences_activity
