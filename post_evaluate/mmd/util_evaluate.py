

import numpy as np
from post_evaluate.mmd.embed import spectrum_map_2, spectrum_Map, embed_DNABert
# from embed import spectrum_map_2, spectrum_Map, embed_DNABert
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel

def mmd_2(arg_dict, seq1=None, seq2=None, emb1=None, emb2=None, kernel_args={}):
    '''
        From Kucera; 
        Calculates MMD between two sets of sequences. 
        Optionally takes embeddings or mean embeddings of sequences if these have been precomputed for efficiency. 
        If <return_pvalue> is true, a Monte-Carlo estimate (1000 iterations) of the p-value is returned. 
        Note that this is compute-intensive and only implemented for the linear kernel.
    '''
    if emb1 is None and emb2 is None:

        if arg_dict["embedding"] not in ["spectrum", "DNABert"]:
            raise NotImplementedError

        if arg_dict["embedding"] == 'spectrum':
            embed = spectrum_map_2
        if arg_dict["embedding"] == 'DNABert':
            embed = embed_DNABert

        emb1, emb2 = embed(arg_dict, seq1, seq2)

    if arg_dict['kernel'] == 'linear':
        x = np.mean(emb1, axis=0)
        y = np.mean(emb2, axis=0)
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))

    elif arg_dict['kernel'] == 'gaussian':
        x = np.array(emb1)
        y = np.array(emb2)
        m = x.shape[0]
        n = y.shape[0]
        Kxx = rbf_kernel(x,x, **kernel_args)#.numpy()
        Kxy = rbf_kernel(x,y, **kernel_args)#.numpy()
        Kyy = rbf_kernel(y,y, **kernel_args)#.numpy()
        MMD = np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )

    if arg_dict['return_pvalue']:
        agg = np.concatenate((emb1,emb2), axis=0)
        mmds = []
        for i in range(1000):
            np.random.shuffle(agg)
            _emb1 = agg[:m]
            _emb2 = agg[m:]
            arg_dict['return_pvalue'] = False
            mmds.append(mmd(arg_dict, emb1=_emb1, emb2=_emb2, kernel_args=kernel_args)[0])
        rank = float(sum([x <= MMD for x in mmds]))+1
        pval = (1000+1-rank)/(1000+1)
        return [MMD, pval]
    else:
        return [MMD]
    

def MMD(seq1=None, seq2=None, Kernel=None, kernel_args={}):
    '''
        From Kucera; 
        Calculates MMD between two sets of sequences. 
        Optionally takes embeddings or mean embeddings of sequences if these have been precomputed for efficiency. 
        If <return_pvalue> is true, a Monte-Carlo estimate (1000 iterations) of the p-value is returned. 
        Note that this is compute-intensive and only implemented for the linear kernel.
    '''
    embed = spectrum_Map
    emb1, emb2 = embed(real_dataset_text=seq1, fake_dataset_text=seq2, mer=3)

    if Kernel == 'linear':
        x = np.mean(emb1, axis=0)
        y = np.mean(emb2, axis=0)
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))

    elif Kernel == 'gaussian':
        x = np.array(emb1)
        y = np.array(emb2)
        m = x.shape[0]
        n = y.shape[0]
        Kxx = rbf_kernel(x,x, **kernel_args)#.numpy()
        Kxy = rbf_kernel(x,y, **kernel_args)#.numpy()
        Kyy = rbf_kernel(y,y, **kernel_args)#.numpy()
        MMD = np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )

    return [MMD]
    


    