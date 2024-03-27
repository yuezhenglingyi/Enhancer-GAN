
import torch
import itertools
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertPreTrainedModel, BertModel, BertTokenizer


class NewsDataset(Dataset):
    def __init__(self, encodings, number):
        self.number = number
        self.encodings = encodings
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return self.number


def embed_DNABert(arg_dict, fake_dataset_text, real_dataset_text):
    """
        parameters:
            for DNABert model:
                mer;
                model_path;
    """
    def Conversion_from_text_to_token(sequences_text, mer, overlapping=True):
        result_sequences = []
        if overlapping:
            pass

        gap = 1 if overlapping else mer
        for sequence in sequences_text:
            content = ""
            for index in range(0, len(sequence)-mer+1, gap):
                token=""
                for i in range(mer):
                    token += sequence[index+i]
                token += " "
                content += token

            result_sequences.append(content)

        return result_sequences


    def Get_embedding_from_DNABert(arg, dataset_text, tokenizer_, DNABert):
        
        dataset_tokens = Conversion_from_text_to_token(dataset_text, arg['mer'])
        # import pdb; pdb.set_trace()
        dataset_encoding = tokenizer_(dataset_tokens, truncation=True, padding = True, max_length = arg['max_length'])
        encoding_dataset = NewsDataset(dataset_encoding, len(dataset_encoding["input_ids"]))
        encoding_dataloader = DataLoader(encoding_dataset, batch_size=arg['batch_size'], shuffle=False)

        embedding_results = []
        for batch_data in encoding_dataloader:
            with torch.no_grad():
                input_ids = batch_data["input_ids"].to(device)
                attention_mask = batch_data["attention_mask"].to(device)

                embeddings_outputs = DNABert(input_ids, attention_mask)

                # only utilize the embedding of CLS
                sequence_CLS = embeddings_outputs[1].cpu().tolist()
                embedding_results.extend(sequence_CLS)
        
        return embedding_results
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(arg_dict['model_path'], output_hidden_states=False) 
    embed_model = BertModel.from_pretrained(arg_dict['model_path']).to(device)

    fake_embeddings = Get_embedding_from_DNABert(arg_dict, fake_dataset_text, tokenizer, embed_model)
    real_embeddings = Get_embedding_from_DNABert(arg_dict, real_dataset_text, tokenizer, embed_model)
    
    fake_results = []
    real_results = []
    for idx, fake_embedding in enumerate(fake_embeddings):
     
        norm = np.sqrt(np.dot(fake_embedding, fake_embedding))
        if norm != 0:
            fake_embedding /= norm
            fake_results.append(fake_embedding)

    for real_embedding in real_embeddings:
        norm = np.sqrt(np.dot(real_embedding, real_embedding))
        if norm != 0:
            real_embedding /= norm
            real_results.append(real_embedding)

    return fake_results, real_results


def spectrum_map_2(arg_dict, fake_dataset_text, real_dataset_text):
    '''
        From: From Kucera;
        Maps a set of sequences to k-mer vector representation.
    '''
    amino_acid_alphabet = "ATCG"
    def make_kmer_trie(kmer):

        kmers = [''.join(i) for i in itertools.product(amino_acid_alphabet, repeat = kmer)]
        kmer_trie = {}
        for i, kmer in enumerate(kmers):
            tmp_trie = kmer_trie
            for aa in kmer:
                if aa not in tmp_trie:
                    tmp_trie[aa] = {}
                if 'kmers' not in tmp_trie[aa]:
                    tmp_trie[aa]['kmers'] = []
                tmp_trie[aa]['kmers'].append(i)
                tmp_trie = tmp_trie[aa]
        return kmer_trie

    trie = make_kmer_trie(arg_dict['mer'])

    def matches(substring):
        d = trie
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['kmers']

    def map(sequence):
        vector = np.zeros(len(amino_acid_alphabet) ** arg_dict['mer'])
        for i in range(len(sequence)-arg_dict['mer']+1):
            for j in matches(sequence[i : i+arg_dict['mer']]):
                if arg_dict['mode'] == 'count':
                    vector[j] += 1
                elif arg_dict['mode'] == 'indicate':
                    vector[j] = 1
        feat = np.array(vector)
        if arg_dict['normalize']:
            norm = np.sqrt(np.dot(feat,feat))
            if norm != 0:
                feat /= norm
        return feat
    
    fake_embedding = np.array([map(seq.replace("\n", "")) for seq in fake_dataset_text], dtype=np.float32).tolist()
    real_embedding = np.array([map(seq.replace("\n", "")) for seq in real_dataset_text], dtype=np.float32).tolist()

    return fake_embedding, real_embedding


def spectrum_Map(mer=3, mode='count', fake_dataset_text=None, real_dataset_text=None):
    '''
        From: From Kucera;
        Maps a set of sequences to k-mer vector representation.
    '''
    amino_acid_alphabet = "ATCG"
    def make_kmer_trie(kmer):

        kmers = [''.join(i) for i in itertools.product(amino_acid_alphabet, repeat = kmer)]
        kmer_trie = {}
        for i, kmer in enumerate(kmers):
            tmp_trie = kmer_trie
            for aa in kmer:
                if aa not in tmp_trie:
                    tmp_trie[aa] = {}
                if 'kmers' not in tmp_trie[aa]:
                    tmp_trie[aa]['kmers'] = []
                tmp_trie[aa]['kmers'].append(i)
                tmp_trie = tmp_trie[aa]
        return kmer_trie

    trie = make_kmer_trie(mer)

    def matches(substring):
        d = trie
        
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['kmers']

    def map(sequence):
        vector = np.zeros(len(amino_acid_alphabet) ** mer)
        for i in range(len((sequence))-mer+1):
            for j in matches(sequence[i:i+mer]):
                if mode == 'count':
                    vector[j] += 1
                elif mode == 'indicate':
                    vector[j] = 1
        feat = np.array(vector)
        if True:
            norm = np.sqrt(np.dot(feat,feat))
            if norm != 0:
                feat /= norm
        return feat

    fake_embedding = np.array([map(seq) for seq in fake_dataset_text], dtype=np.float32).tolist()
    real_embedding = np.array([map(seq) for seq in real_dataset_text], dtype=np.float32).tolist()

    return fake_embedding, real_embedding



