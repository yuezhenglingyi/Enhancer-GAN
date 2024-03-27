
import os
import numpy as np
import pandas as pd
from util_evaluate import mmd_2


args_dict={    
            "mer": 3,
            "embedding": "spectrum", 
            "max_length": 250,
            "batch_size": 256,
            
            "mode": "count",
            "normalize": True,
            "kernel": "linear",
            "return_pvalue": False,

            "model_path": "/3-new-12w-0"
            }


def Get_Seqence_Activity(data_file):
    seq_list = []
    act_list = []
    with open(data_file, "r") as f:
        contents = f.readlines()
        for content in contents[1:]:
            items = content.split("\t")
            seq = items[0]
            items = items[1].replace("\n", "").replace("[", "").replace("]", "")

            seq_list.append(seq)
            act_list.append(float(items))
    
    return seq_list, act_list

def Get_Seqence(data_file):
    seq_list = []
    with open(data_file, "r") as f:
        contents = f.readlines()
        for content in contents[1:]:
            items = content.split("\n")
            seq = items[0]
            seq_list.append(seq)

    return seq_list

def Get_Seqence_2(data_file):
    seq_list = []
    with open(data_file, "r") as f:
        contents = f.readlines()
        contents = contents[0]
        for i in range(0, 20000):
            seq = ""
            for j in range(i*249, (i+1)*249):
                seq += contents[j]
            seq_list.append(seq)

    return seq_list


def Calculate_mmd(data_file, threshold=0.0, gap_low_boundary=100.0, gap_high_boundary=-100.0):
    act_list = []
    seq_list = []
    with open(data_file, "r") as f:
        contents = f.readlines()
        for content in contents[1:]:
            score = content.split("\t")[1].replace("\n", "").replace("[", "").replace("]", "")
            seq = content.split("\t")[0]
            act_list.append(float(score))
            seq_list.append(seq)
    
    low_act_seqs = []
    high_act_seqs = []
    
    if gap_low_boundary > gap_high_boundary:
        gap_low_boundary = threshold
        gap_high_boundary = threshold
    
    for index, act in enumerate(act_list):
        if act >= gap_high_boundary:
            high_act_seqs.append(seq_list[index])
        elif act <= gap_low_boundary:
            low_act_seqs.append(seq_list[index])
        
    mmd_value = mmd_2(args_dict, low_act_seqs, high_act_seqs)[0]
    return mmd_value, len(low_act_seqs), len(high_act_seqs)


if __name__ == "__main__":
    record_file = "record_2.txt"
    fw = open(record_file, 'a')

    train_file_path = "../../data/fake_activity_Sequence_train.txt"
    train_sequences, _ = Get_Seqence_Activity(train_file_path)

    test_file_path = "../../checkpoint/Vanilla-GAN/1689393684/samples"
    test_sequences = Get_Seqence_2(test_file_path)

    for emb in ["spectrum", "DNABert"]:
        for mer in [3,4,5,6]:
            args_dict["mer"] = mer
            args_dict["embedding"] = emb
            args_dict["model_path"] = "/{}-new-12w-0".format(mer)

            mmd_sum = 0.0
            for idx in range(1, 6, 1):

                indexes = set(np.random.randint(0, len(train_sequences), size=23000))
                sequences_one = np.array(train_sequences)[list(indexes)]

                mmd_value = mmd_2(args_dict, test_sequences, sequences_one)[0]
                content = "train(2w) vs origin wgan_gp (epoch=91); {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
                mmd_sum += mmd_value
                fw.write(content)
                fw.flush()
            content = "--> average: {:.5f}\n".format(mmd_sum/5)
            fw.write(content)
            fw.flush()
    
    fw.close()