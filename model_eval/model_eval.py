import os
import numpy as np

import sys
sys.path.append("../")
from a_Vanilla_GAN import WGAN_LangGP

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def  getSequence(sequence_path, number):
    sequence_file = open(sequence_path, "r")
    sequences = sequence_file.readlines()
    sequences = [sequence.replace("\n", "") for sequence in sequences]
    index_set = set()
    while len(index_set) < number:
        index = np.random.randint(len(sequences))
        index_set.add(index)
    indexes = list(index_set)

    sequences = np.array(sequences)[indexes].tolist()
    return sequences


if __name__ == "__main__":
    
    number = 20000
    iterations = []
    real_sequence_path = "../data/Sequence_train.txt"
    model_path = "../checkpoint/Vanilla-GAN/1689393684"
    real_sequences = getSequence(real_sequence_path, number)

    for i in range(70):
        iterations.append(i*1000 + 999)
    
    model = WGAN_LangGP()
    for iteration in iterations:
        fake_sequences = []

        model.load_model(directory=model_path, iteration=iteration)
        n_batch = int(number/model.batch_size)
        for batch in range(n_batch):
            seqs = model.sample_generator(model.batch_size).to("cpu").tolist()
            fake_sequences.extend(seqs)


