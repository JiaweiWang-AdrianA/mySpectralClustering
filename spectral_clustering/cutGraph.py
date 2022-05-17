import numpy as np
from sklearn.cluster import KMeans


def calLaplacianMatrix_RCUT(simMatrix):
    ''' calculate Laplacian Matrix using RatioCut
        output: Laplacian Matrix
    '''
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(simMatrix, axis=1)
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - simMatrix #np.diag:以一维数组的形式返回方阵的对角线
    # print degreeMatrix
    # pirnt(degreeMatrix)
    return np.nan_to_num(laplacianMatrix)


def calLaplacianMatrix_NCUT(simMatrix):
    ''' calculate Laplacian Matrix using NCut
        output: Normalized Laplacian Matrix
    '''
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(simMatrix, axis=1)
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - simMatrix #np.diag:以一维数组的形式返回方阵的对角线
    # print degreeMatrix
    # pirnt(degreeMatrix)
    # normailze: n_L = D^(-1/2) L D^(-1/2)
    np.seterr(divide='ignore', invalid='ignore')
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    n_laplacianMatrix = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    return np.nan_to_num(n_laplacianMatrix)



def calOptimalIndicatorByKMeans(simMatrix, cut_method='RCut', cluster_num=2, k=None):
    ''' calculate Optimal Indicator Vector by k-means
        output: optimal cut solution
    '''
    if cut_method == 'RCut':
        Laplacian = calLaplacianMatrix_RCUT(simMatrix)
    else:
        Laplacian = calLaplacianMatrix_NCUT(simMatrix)
    if not k:
        k = cluster_num

    # 计算方形矩阵Laplacian的特征值和特征向量
    # x多个特征值组成的一个矢量
    # V多个特征向量组成的一个矩阵。
    # 每一个特征向量都被归一化了。
    # 第i列的特征向量v[:,i]对应第i个特征值x[i]。
    x, V = np.linalg.eig(Laplacian)
    # 特征值与序号组成列表
    x = zip(x, range(len(x)))
    # 按列表的第一个元素(特征值)升序排列
    x = sorted(x, key=lambda x: x[0])
    # 取最小的k个特征值对应的特征行向量
    # 按垂直方向（行顺序）堆叠数组构成一个新的数组,然后转置
    H = np.vstack([V[:, i] for (v, i) in x[:k]]).T
    if(isinstance(H[0][0],complex)): #判断是否虚数
        H = abs(H)
    optH_kmeans = KMeans(n_clusters=cluster_num).fit(H)
    return optH_kmeans


# def calCost(simMatrix, H, cut_method, cluster_num):
#     N = len(simMatrix)
#     cost = [0]*cluster_num
#     if cut_method == 'RCut':
#         cls_cnt = [0]*cluster_num
#         for i in range(N):
#             for j in range(N):
#                 k_i, k_j, s_ij = int(H[i]), int(H[j]), simMatrix[i][j]
#                 cls_cnt[k_i], cls_cnt[k_j] = cls_cnt[k_i]+1, cls_cnt[k_j]+1
#                 if k_i!=k_j: 
#                     cost[k_i], cost[k_j] = cost[k_i]+s_ij, cost[k_j]+s_ij
#         for k in range(cluster_num):
#             if cost[k]: cost[k] /= cls_cnt[k]
#     else:
#         cls_d = [0]*cluster_num
#         for i in range(N):
#             for j in range(N):
#                 k_i, k_j, s_ij = int(H[i]), int(H[j]), simMatrix[i][j]
#                 cls_d[k_i], cls_d[k_j] = cls_d[k_i]+s_ij, cls_d[k_j]+s_ij
#                 if k_i!=k_j: 
#                     cost[k_i], cost[k_j] = cost[k_i]+s_ij, cost[k_j]+s_ij
#         for k in range(cluster_num):
#             if cost[k]: cost[k] /= cls_d[k]
#     return sum(cost)


# def calOptimalIndicatorByBrute(simMatrix, cut_method='RCut', cluster_num=2):
#     ''' calculate Optimal Indicator Vector by brute
#         output: optimal cut solution
#     '''
#     N = len(simMatrix)
#     H = np.zeros(N)
#     min_cost, optH = 0, np.zeros(N)*-1
#     j = N-1
#     while(j>=0):
#         for i in range(j+1, N):
#             for k in range(cluster_num):
#                 H[i]= k
#                 cost = calCost(simMatrix, H, cut_method, cluster_num)
#                 if min_cost <= 0 or (cost > 0 and min_cost > cost):
#                     optH, min_cost = H, cost
#                     # print(optH)
#                     # print(min_cost)
#         if H[j]<cluster_num-1:
#             H[j]+=1
#         else: 
#             j-=1
#     return optH




