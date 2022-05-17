import sys
import numpy as np
import random
from itertools import cycle, islice
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import title


def euclidDistance(x1, x2, sqrt_flag=False):
    ''' Euclidean distance '''
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def jaccardSimilarity(s1, s2):
    ''' Jaccard Similarity
        input: s1(set/list/tuple), s2(set/list/tuple)
        output: Jaccard Similarity between s1 and s2
    '''
    len_union = len(set(s1).union(set(s2)))
    len_inter = len(set(s1)-(set(s1)-set(s2)))
    if len_union != 0:
        res = len_inter/len_union 
    else:
        res = 0
    return res


def plotTestRes(X, y, title, dataset_name):
    sys.path.append("..")
    colors = np.array(list(islice(
        cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']),
        int(max(y) + 1))))
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    plt.savefig("./figs/"+str(dataset_name)+'/'+str(title)+".png")
    plt.close('all')


def genGraphFromAdjacent(Adjacent, matInx2bIds=None):
    G = nx.Graph()
    N = len(Adjacent)
    for i in range(N):
        for j in range(i+1, N):
            (bid_i,bid_j) = (matInx2bIds[i],matInx2bIds[j]) if matInx2bIds else (i,j)
            G.add_node(bid_i)
            G.add_node(bid_j)
            weight = Adjacent[i][j]
            if weight > 0:
                G.add_edge(bid_i, bid_j, weight=weight)

    return G


def draw_graph(G, weight_edge_flag=False, save_file_path='Graph.png', pos=None):
    """ generate a viewable view of the graph """
    print('drawing the graph...')
    plt.rcParams['figure.figsize']= (20, 20)
    # positions for all nodes
    if not pos:
        pos = nx.shell_layout(G) 
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=1000, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='white', font_weight='bold')
    # draw edges
    if not weight_edge_flag:
        nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=0.8, edge_color='red', alpha=0.7)
    else:
        # weights for all edges
        nx.draw_networkx_edges(G, pos, width=[float(d['weight']*10) for (u,v,d) in G.edges(data=True)], edge_color='red', alpha=0.7)
        weights = nx.get_edge_attributes(G, 'weight')
        # round weights
        for edge in weights.keys():
            weights[edge] = round(weights[edge],2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=14)
        
    plt.savefig(save_file_path, dpi=200)
    print('drawing done. The drawing is saved in [' + save_file_path +']')
    #plt.show()
    plt.close('all')
    return pos


def draw_Graph_with_clustering(G, bid2cid, weight_edge_flag=False, save_file_path='Graph_res.png', pos=None):
    """ generate a viewable view of the graph """
    print('drawing the graph...')
    plt.rcParams['figure.figsize']= (20, 20)

    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    # positions for all nodes
    if not pos:
        pos = nx.shell_layout(G) 
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=[colors[bid2cid[bid]] for bid in G.nodes()], alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='white', font_weight='bold')
    # draw edges
    if not weight_edge_flag:
        nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=0.8, edge_color='black', alpha=0.7)
    else:
        # weights for all edges
        # nx.draw_networkx_edges(G, pos, width=[float(d['weight']*10) for (u,v,d) in G.edges(data=True)], edge_color='black', alpha=0.7)
        e_lables, del_edges, sav_edges = {}, [], []
        for edge in G.edges(data=True):
            di,dj = edge[0],edge[1]
            if bid2cid[di] != bid2cid[dj]: 
                del_edges.append(edge)
            else:
                e_lables[(di,dj)] = round(edge[2]['weight'],2)
                sav_edges.append(edge)
        nx.draw_networkx_edges(G, pos, edgelist=sav_edges,  width=[float(d['weight']*10) for (u,v,d) in sav_edges], edge_color='black', alpha=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=del_edges, width=2, edge_color='red', style='dashed', alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=e_lables, font_size=14)
        
    plt.savefig(save_file_path, dpi=200)
    print('drawing done. The drawing is saved in [' + save_file_path +']')
    #plt.show()
    plt.close('all')
    return pos, del_edges
    




