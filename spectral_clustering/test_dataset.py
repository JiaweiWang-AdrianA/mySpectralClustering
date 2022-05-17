from utils import *
from getDataset import *
from getGraph import *
from cutGraph import *


def test(data, dataset_name='data_1', cluster_num=2, sim_method='KNN', sigma=0.2, param=None, cut_method='NCut', kmeans_k=None):
    # Spectral Clustering
    # Generate Graph 
    # get Similarity Matrix
    S_g = calGaussianSimMatrixFromData(data, sigma=sigma)
    print("S_g:", S_g)
    S = genSimMatrixFromSimOrgMatrix(S_g, method=sim_method, param=param)
    print("S:", S)

    # Cut Graph
    sp_kmeans = calOptimalIndicatorByKMeans(S, cut_method=cut_method, cluster_num=cluster_num, k=kmeans_k)
    #sp_kmeans = calOptimalIndicatorByBrute(S, cut_method=cut_method, cluster_num=cluster_num)
    
    
    # plot the results
    name = 'sc_'+str(sim_method)+'_'+str(cut_method)+'_sigma'+str(sigma)
    plotTestRes(data, sp_kmeans.labels_, name, dataset_name)

    # # get Graph
    # G = genGraphFromAdjacent(S)
    # draw_graph(G, weight_edge_flag=True)

 
if __name__ == '__main__':
    # # seed( ) 用于指定随机数生成时所用算法开始的整数值。
    # # 1.如果使用相同的seed( )值，则每次生成的随即数都相同；
    # # 2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    # # 3.设置的seed()值仅一次有效
    # np.random.seed(1)
    cluster_num = 2
    # Get Dataset
    data1, label1 = genTwoCircles(n_samples=200) #返回data(生成的样本),label(每个样本的类成员的整数标签0或1)
    data2, label2 = genMoons(n_samples=200)
    # plot the correct result
    plotTestRes(data1, label1, 'real_labels', 'data_1')
    plotTestRes(data2, label2, 'real_labels', 'data_2')

    # plot the kmeans result
    pure_kmeans1 = KMeans(n_clusters=cluster_num).fit(data1)
    plotTestRes(data1, pure_kmeans1.labels_, 'kmeans', 'data_1')
    pure_kmeans2 = KMeans(n_clusters=cluster_num).fit(data2)
    plotTestRes(data2, pure_kmeans2.labels_, 'kmeans', 'data_2')

    # testing
    for sigma in [0.1, 0.5, 1]:
        for sim_method in ['Full', 'ENbrs', 'KNN']:
            for cut_method in ['RCut', 'NCut']:
                test(data1, 'data_1', sim_method=sim_method, cut_method=cut_method, cluster_num=cluster_num, sigma=sigma)
                test(data2, 'data_2', sim_method=sim_method, cut_method=cut_method, cluster_num=cluster_num, sigma=sigma)





