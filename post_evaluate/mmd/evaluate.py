import os
import json
import argparse

from post_evaluate.mmd.util_evaluate import mmd_2

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def Args_dict_from_json(json_path):
    args = dict()
    with open(json_path) as json_file:
        args_json = json.load(json_file)

    for key,value in args_json.items():
        args[key] = value
    
    return args


def Get_Sequence_from_file(file_path):
    sequences = []
    sequence_file = open(file_path, "r")
    contents = sequence_file.readlines()
    for content in contents:
        if "\t" in content:
            sequences.append(content.split("\t")[0])
        else:
            sequences.append(content.replace("\n", ""))
    return sequences
    

if __name__ == "__main__":
    json_path = "config_evaluate.json"
    args = Args_dict_from_json(json_path)
    
    real_sequences = Get_Sequence_from_file(args["real_file_path"])
    fake_sequences = Get_Sequence_from_file(args["fake_file_path"])

    mmd_result = mmd_2(args, real_sequences, fake_sequences)
    """
        "embedding": "spectrum", "DNABert"
        "kernel": "linear", "gaussian"
    """
    print("the result of MMD:{}".format(mmd_result))
