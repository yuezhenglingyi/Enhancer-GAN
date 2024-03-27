
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


def Differenate_Act(sequences, acts, low_threshlod, high_threshlod ):
    low_sequences = []
    high_sequences = []
    for idx in range(len(sequences)):
        sequence = sequences[idx]
        act = acts[idx]
        if act < low_threshlod:
            low_sequences.append(sequence)
        elif act >= high_threshlod:
            high_sequences.append(sequence)
    
    return low_sequences, high_sequences

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
    # file_path = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/real_Sequence_activity_train.txt"

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.0, gap_high_boundary=0.0)
    # print("low=-0.0, high=0.0; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.5, gap_high_boundary=0.5)
    # print("low=-0.5, high=0.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-0.5, high=0.5; mmd: 0.10284666219086874; num_low: 59902, num_high: 86294

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-1.0, gap_high_boundary=1.5)
    # print("low=-1.5, high=1.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-1.5, high=1.5; mmd: 0.12293952977974998; num_low: 34360, num_high: 40183

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-2.0, gap_high_boundary=2.5)
    # print("low=-2.0, high=2.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-2.0, high=2.5; mmd: 0.16090727646803207; num_low: 8341, num_high: 17434

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-2.5, gap_high_boundary=3.5)
    # print("low=-2.5, high=3.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-2.5, high=3.5; mmd: 0.17968215476147484; num_low: 3993, num_high: 6918

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-2.5, gap_high_boundary=4.0)
    # print("low=-2.5, high=4.0; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-2.5, high=4.0; mmd: 0.18326076099221872; num_low: 3993, num_high: 4005

    # file_path = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/fake_activity_Sequence_train.txt"
    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.5, gap_high_boundary=0.5)
    # print("low=-0.5, high=0.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-0.5, high=0.5; mmd: 0.21570729193453525; num_low: 38403, num_high: 73422

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-1.0, gap_high_boundary=1.5)
    # print("low=-1.5, high=1.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-1.5, high=1.5; mmd: 0.4509955931094975; num_low: 7426, num_high: 32099

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-1.0, gap_high_boundary=2.5)
    # print("low=-1.0, high=2.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-1.0, high=2.5; mmd: 0.45685654790040225; num_low: 7426, num_high: 10369

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-1.0, gap_high_boundary=3.5)
    # print("low=-1.0, high=3.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=-1.0, high=3.5; mmd: 0.46513677727618485; num_low: 7426, num_high: 2699

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=0.0, gap_high_boundary=0.0)
    # print("low=0.0, high=0.0; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))
    # # low=0.0, high=0.0; mmd: 0.09337337251340311; num_low: 95722, num_high: 105417
    
    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.0, gap_high_boundary=0.5)
    # print("low=-0.0, high=0.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.0, gap_high_boundary=1.5)
    # print("low=-0.0, high=1.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.0, gap_high_boundary=2.5)
    # print("low=-0.0, high=1.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))

    # mmd, num_low, num_high = Calculate_mmd(file_path, gap_low_boundary=-0.0, gap_high_boundary=3.5)
    # print("low=-0.0, high=1.5; mmd: {}; num_low: {}, num_high: {}".format(mmd, num_low, num_high))

    # val_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/fake_activity_Sequence_val.txt"
    # test_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/fake_activity_Sequence_test.txt"

    # val_sequences, val_acts = Get_Seqence_Activity(val_filename)
    # test_sequences, test_acts = Get_Seqence_Activity(test_filename)

    # thresholds = [-100, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    # for threshold in thresholds:
    #     sequences = []
    #     for idx, seq in enumerate(test_sequences):
    #         if test_acts[idx] > threshold:
    #             sequences.append(test_sequences[idx])
    #     mmd_value = mmd_2(args_dict, sequences, val_sequences)[0]
    #     print("mmd: {:.4f}; test act>{}; number of test: {}; number of val:{};".format(mmd_value, threshold, len(sequences), len(val_sequences)))
    test_file_path = "../../data/fake_activity_Sequence_test.txt"
    val_file_path = "../../data/fake_activity_Sequence_val.txt"


    test_sequences, _ = Get_Seqence_Activity(test_file_path)
    val_sequences, _ = Get_Seqence_Activity(val_file_path)

    record_file = "record.txt"
    fw = open(record_file, 'a')

    # args_dict["mer"] = 3
    # args_dict["embedding"] = "spectrum"
    # mmd_mer_3 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # args_dict["mer"] = 4
    # args_dict["embedding"] = "spectrum"
    # mmd_mer_4 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # args_dict["mer"] = 5
    # args_dict["embedding"] = "spectrum"
    # mmd_mer_5 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # content_1 = "test vs val; spectrum; mmd_mer_3: {:.5f}".format(mmd_mer_3)
    # content_2 = "test vs val; spectrum; mmd_mer_4: {:.5f}".format(mmd_mer_4)
    # content_3 = "test vs val; spectrum; mmd_mer_5: {:.5f}".format(mmd_mer_5)

    # print(content_1)
    # print(content_2)
    # print(content_3)

    # fw.write(content_1+"\n")
    # fw.write(content_2+"\n")
    # fw.write(content_3+"\n")
    # fw.flush()

    # args_dict["mer"] = 3
    # args_dict["embedding"] = "DNABert"
    # args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/3-new-12w-0"
    # mmd_mer_3 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # args_dict["mer"] = 4
    # args_dict["embedding"] = "DNABert"
    # args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/4-new-12w-0"
    # mmd_mer_4 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # args_dict["mer"] = 5
    # args_dict["embedding"] = "DNABert"
    # args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/5-new-12w-0"
    # mmd_mer_5 = mmd_2(args_dict, test_sequences, val_sequences)[0]
    # content_1 = "test vs val; DNABert; mmd_mer_3: {:.5f}".format(mmd_mer_3)
    # content_2 = "test vs val; DNABert; mmd_mer_3: {:.5f}".format(mmd_mer_4)
    # content_3 = "test vs val; DNABert; mmd_mer_3: {:.5f}".format(mmd_mer_5)

    # print(content_1)
    # print(content_2)
    # print(content_3)

    # fw.write(content_1+"\n")
    # fw.write(content_2+"\n")
    # fw.write(content_3+"\n")
    # fw.flush()

    # record_file = "record.txt"
    # fw = open(record_file, 'a')

    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):
    #             special_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/Random_seqs/special_seqs_21k/special_seqs_21k_{}.fa".format(idx)
    #             special_seqs = Get_Seqence(special_filename)
    #             mmd_value = mmd_2(args_dict, special_seqs, test_sequences)[0]
    #             content = "test vs random_special; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()

    # fw.write("\n")
    # fw.flush()
    
    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):
    #             uniform_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/Random_seqs/uniform_seqs_21k/uniform_seqs_21k_{}.fa".format(idx)
    #             uniform_seqs = Get_Seqence(uniform_filename)
    #             mmd_value = mmd_2(args_dict, uniform_seqs, test_sequences)[0]
    #             content = "test vs random_uniform; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()

    # fw.write("\n")
    # fw.flush()

    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):
    #             special_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/Random_seqs/special_seqs_21k/special_seqs_21k_{}.fa".format(idx)
    #             special_seqs = Get_Seqence(special_filename)
    #             mmd_value = mmd_2(args_dict, special_seqs, val_sequences)[0]
    #             content = "val vs random_special; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()

    # fw.write("\n")
    # fw.flush()

    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):
    #             uniform_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/Random_seqs/uniform_seqs_21k/uniform_seqs_21k_{}.fa".format(idx)
    #             uniform_seqs = Get_Seqence(uniform_filename)
    #             mmd_value = mmd_2(args_dict, uniform_seqs, val_sequences)[0]
    #             content = "val vs random_uniform; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()

    # fw.write("\n")
    # fw.flush()

    # train_file_path = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/fake_activity_Sequence_train.txt"
    # train_sequences, _ = Get_Seqence_Activity(train_file_path)

    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):

    #             indexes = set(np.random.randint(0, len(train_sequences), size=23000))
    #             sequences_one = np.array(train_sequences)[list(indexes)]

    #             mmd_value = mmd_2(args_dict, test_sequences, sequences_one)[0]
    #             content = "train(2w) vs test; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()


    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):

    #             indexes = set(np.random.randint(0, len(train_sequences), size=23000))
    #             sequences_one = np.array(train_sequences)[list(indexes)]

    #             mmd_value = mmd_2(args_dict, val_sequences, sequences_one)[0]
    #             content = "train(2w) vs val sequences; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()


    # fake_prefix = "/geniusland/home/xiaoliwei/fbgan-from-colab/checkpoint/FBGAN_AMP_lijiahao/1689393684/samples"
    # fake_filenames = ["sampled_198.txt", "sampled_199.txt", "sampled_200.txt"]
    # for fake_filename in fake_filenames:
    #     filename = os.path.join(fake_prefix, fake_filename)
    #     fake_sequences, _ = Get_Seqence_Activity(filename)
    #     epoch = fake_filename[8:11]
    #     for emb in ["spectrum", "DNABert"]:
    #         for mer in [3,4,5,6]:
    #             args_dict["mer"] = mer
    #             args_dict["embedding"] = emb
    #             args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #             mmd_sum = 0.0
    #             for idx in range(1, 6, 1):

    #                 indexes = set(np.random.randint(0, len(train_sequences), size=23000))
    #                 sequences_one = np.array(train_sequences)[list(indexes)]

    #                 mmd_value = mmd_2(args_dict, fake_sequences, sequences_one)[0]
    #                 content = "train(2w) vs fake seqs (epoch={}); {}; iter={}; mmd_{}: {:.5f}\n".format(epoch, emb, idx, mer, mmd_value)
    #                 mmd_sum += mmd_value
    #                 fw.write(content)
    #                 fw.flush()
    #             content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #             fw.write(content)
    #             fw.flush()


    # fw.write("\n")
    # fw.flush()


    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5,6]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)

    #         mmd_sum = 0.0
    #         for idx in range(1, 6, 1):
    #             uniform_filename = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/Random_seqs/uniform_seqs_21k/uniform_seqs_21k_{}.fa".format(idx)
    #             uniform_seqs = Get_Seqence(uniform_filename)

    #             indexes = set(np.random.randint(0, len(train_sequences), size=23000))
    #             sequences_one = np.array(train_sequences)[list(indexes)]

    #             mmd_value = mmd_2(args_dict, uniform_seqs, sequences_one)[0]
    #             content = "train(2w) vs random_uniform; {}; iter={}; mmd_{}: {:.5f}\n".format(emb, idx, mer, mmd_value)
    #             mmd_sum += mmd_value
    #             fw.write(content)
    #             fw.flush()
    #         content = "--> average: {:.5f}\n".format(mmd_sum/5)
    #         fw.write(content)
    #         fw.flush()

    # fw.write("\n")
    # fw.flush()


    # # import pdb; pdb.set_trace()
    # train_file_path = "/geniusland/home/xiaoliwei/fbgan-from-colab/data/DeepSTARR/fake_activity_Sequence_train.txt"
    # sequences, _ = Get_Seqence_Activity(train_file_path)

    # for emb in ["spectrum", "DNABert"]:
    #     for mer in [3,4,5]:
    #         args_dict["mer"] = mer
    #         args_dict["embedding"] = emb
    #         args_dict["model_path"] = "/geniusland/home/xiaoliwei/fbgan-from-colab/post_evaluate/mmd/{}-new-12w-0".format(mer)
    #         mmd_list = []

    #         for iter_ in range(5):
    #             indexes_one_ = set(np.random.randint(0, len(sequences), size=100000))
    #             indexes_two_ = set(np.random.randint(0, len(sequences), size=100000))
    #             indexes_one = list(indexes_one_ - indexes_two_)
    #             indexes_two = list(indexes_two_ - indexes_one_)
    #             sequences_one = np.array(sequences)[indexes_one]
    #             sequences_two = np.array(sequences)[indexes_two]

    #             mmd_value = mmd_2(args_dict, sequences_one, sequences_two)[0]
    #             mmd_list.append(mmd_value)
    #             content = "{}; mer-{}, iter={}; mmd between train: {:.5f}".format(emb, mer, iter_+1, mmd_value)
    #             fw.write(content+"\n")
    #             print(content)

    #         content = "{}; mer-{}, average: {:.5f}".format(emb, mer, np.mean(mmd_list))
    #         fw.write(content+"\n")
    #         fw.flush()
    #         print(content)

    train_file_path = "../../data/fake_activity_Sequence_train.txt"
    train_sequences, train_acts = Get_Seqence_Activity(train_file_path)

    # low_threshlods = [-0.5, -1.0, -1.0, -1.0, -1.0]
    # high_threshlods = [0.5, 1.5, 2.5, 3.0, 3.5]

    low_threshlods = [0.0, 0.0]
    high_threshlods = [3.0, 3.5]
    
    for idx in range(len(high_threshlods)):
        low_threshlod = low_threshlods[idx]
        high_threshlod = high_threshlods[idx]
        low_sequences, high_sequences = Differenate_Act(train_sequences, train_acts, low_threshlod, high_threshlod )

        for emb in ["spectrum"]:
            for mer in [3,4,5,6]:
                args_dict["mer"] = mer
                args_dict["embedding"] = emb
                args_dict["model_path"] = "{}-new-12w-0".format(mer)

                mmd_value = mmd_2(args_dict, low_sequences, high_sequences)[0]
                content = "low({:.4f}) vs high({:.4f}); {}; mmd_{}: {:.5f}\n".format(low_threshlod, high_threshlod, emb, mer, mmd_value)
                print(content[:-1])
                fw.write(content)
                fw.flush()
               
    fw.close()  
