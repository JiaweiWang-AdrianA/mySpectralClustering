import sys
import numpy as np
import networkx as nx
from utils import *



def calSimMatrixFromBlocks(blocks):
    ''' get the Similarity Matrix from blocks '''
    blocks_num = len(blocks)
    S, matInx2bIds = np.ones((blocks_num, blocks_num)), {}
    bids = list(blocks.keys())
    for i in range(blocks_num):
        for j in range(i+1, blocks_num):
            bid_i, bid_j = bids[i], bids[j]
            matInx2bIds[i], matInx2bIds[j]  = bid_i, bid_j
            block_i, block_j = blocks[bid_i], blocks[bid_j]
            S[i][j] = jaccardSimilarity(block_i, block_j)
            S[j][i] = S[i][j]
    return S, matInx2bIds


def calGaussianSimMatrixFromData(data, sigma=0.1):
    '''
        get the Gaussian Similarity Matrix of dataset
        output: Gaussian Similarity Matrix
    '''
    X = np.array(data)
    S_g = np.ones((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            eucDis = euclidDistance(X[i], X[j])
            S_g[i][j] = np.exp(-eucDis/2/sigma/sigma)
            S_g[j][i] = S_g[i][j]
    return S_g


def genSimMatrixFromSimOrgMatrix(S_org, method='KNN', param=None):
    ''' generate similarity matrix of graph from original matrix by different methods
        output: similarity matrix of graph
    '''
    N = len(S_org)
    S = np.zeros((N,N))
    # Full Connection
    if method == 'Full':
        S = S_org
    # E-Neighbors
    elif method == 'ENbrs':
        e = param if param else 0.5
        for i in range(N):
            for j in range(i, N):
                S[i][j] = e if S_org[i][j] >= e else 0
                S[j][i] = S[i][j]
    # KNN
    else: 
        k = param if param else 10
        for i in range(N):
            dist_with_index = zip(S_org[i], range(N))
            dist_with_index = sorted(dist_with_index, key=lambda x:x[0], reverse=True)
            nbrs_id = [dist_with_index[m][1] for m in range(min(k+1,N))] 
            print(str(i)+':'+str(nbrs_id))
            for j in nbrs_id:
                S[i][j] = S_org[i][j] if i != j else 0
        # Averaging to generate symmetric matrix
        S = (S+S.T)/2
        #print(S)
    return S


def genGraphFromSimMatrix(S, matInx2bIds=None):
    ''' generate the nx graph from similarity matrix '''
    G = nx.Graph()
    N = len(S)
    for i in range(N):
        for j in range(i+1, N):
            (bid_i,bid_j) = (matInx2bIds[i],matInx2bIds[j]) if matInx2bIds else (i,j)
            weight = S[i][j]
            if weight > 0:
                G.add_edge(bid_i, bid_j, weight=weight)
    return G



    




