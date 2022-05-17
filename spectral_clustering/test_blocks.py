from utils import *
from getDataset import *
from getGraph import *
from cutGraph import *

global matInx2bIds,bid2cid,cid2bids

def testSimMethod(blocks, dataset_name='data_blocks', sim_method='Full', param=None):
    global matInx2bIds
    # Generate Graph 
    # get Similarity Matrix and matInx2bIds Dictionary
    S_g, matInx2bIds = calSimMatrixFromBlocks(blocks)
    #print("S_g:", S_g)
    S = genSimMatrixFromSimOrgMatrix(S_g, method=sim_method, param=param)
    #print("S:", S)
    # plot the Graph
    save_path = './figs/' + dataset_name + '/sc_'+str(sim_method)+'.png'
    G = genGraphFromAdjacent(S, matInx2bIds=matInx2bIds)
    G_pos = draw_graph(G, weight_edge_flag=True, save_file_path=save_path)
    return S,G,G_pos


def testCutMethod(SimMatrix, G, G_pos, save_path='figs/data_blocks/sc_Full', cluster_num=2, cut_method='NCut', kmeans_k=None):
    global matInx2bIds,bid2cid,cid2bids
    # Cut Graph
    sp_kmeans = calOptimalIndicatorByKMeans(SimMatrix, cut_method=cut_method, cluster_num=cluster_num, k=kmeans_k)
    bid2cid,cid2bids = {},defaultdict(list)
    for i in range(len(SimMatrix)):
        bid = matInx2bIds[i]
        cid = sp_kmeans.labels_[i]
        bid2cid[bid] = cid
        cid2bids[cid].append(bid)
    #print(bid2cid)
    print(" cluster result: ", list(cid2bids.values()))
    # plot the results
    save_path += '_'+str(cut_method)+'.png'
    draw_Graph_with_clustering(G, bid2cid, pos=G_pos, weight_edge_flag=True, save_file_path=save_path)
    return sp_kmeans

 
if __name__ == '__main__':
    cluster_num = 9
    # Get blocks
    print('----------------- block dataset -----------------')
    blocks = genBlocks(products_num=100,max_block=5,blocks_num=20)
    for bid,block in blocks.items():
        print(str(bid) + " : " + str(block))
    print('-------------- spectral clustering --------------')
    
    # testing
    for sim_method in ['Full', 'ENbrs', 'KNN']:
        print('\n -- sim_method:' + sim_method + ' --')
        if sim_method == 'ENbrs':
            S, G, G_pos = testSimMethod(blocks, sim_method=sim_method, param=0.1)
        elif sim_method == 'KNN':
            S, G, G_pos = testSimMethod(blocks, sim_method=sim_method, param=3)
        else:
            S, G, G_pos = testSimMethod(blocks, sim_method=sim_method)
        for cut_method in ['RCut', 'NCut']:
            print(' [ cut_method:' + cut_method + ' ]')
            save_path = 'figs/data_blocks/sc_'+str(sim_method)
            testCutMethod(S, G, G_pos, save_path=save_path, cluster_num=cluster_num, cut_method=cut_method)





