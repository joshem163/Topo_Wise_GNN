import networkx as nx
from networkx import ego_graph
import numpy as np
import pandas as pd
import pyflagser
from data_loader import *
def Average(lst):
    # average function
    avg = np.average(lst)
    return (avg)


def Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    return intersection


def Degree_list(Graph):
    degree_list = [Graph.degree(node) for node in Graph.nodes]
    return np.array(degree_list)


def wise_embeddings(dataset_Name):
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    basis = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]
    sel_basis = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * 0.1))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]

    Fec = []
    SFec = []

    for i in range(Number_nodes):
        vec = []
        Svec = []

        # Extract the features for the current node
        f = Data.loc[i, feature_names].values.flatten().tolist()

        # Compute similarities for basis
        for b in basis:
            vec.append(Similarity(f, b))

        # Compute similarities for sel_basis
        for sb in sel_basis:
            Svec.append(Similarity(f, sb))

        # Clear the feature list and append results
        f.clear()
        Fec.append(vec)
        SFec.append(Svec)

    return Fec, SFec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def ContextualPubmed(dataset_Name):
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
    Domain_Fec = pd.DataFrame(data.x.numpy())

    # Scale data before applying PCA
    scaling = StandardScaler()

    # Use fit and transform method
    scaling.fit(Domain_Fec)
    Scaled_data = scaling.transform(Domain_Fec)

    # Set the n_components=3
    m = 100
    principal = PCA(n_components=m)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)
    return x

def wise_embeddings_eucledian(dataset_Name):
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    sel_basis = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    Fec = []
    for i in range(Number_nodes):
        #print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        vec = []
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(sel_basis[j])))
        f.clear()
        Fec.append(vec)
    return Fec


def Topological_Feature_subLevel(adj, filtration_fun, Filtration):
    betti_0 = []
    betti_1 = []
    for p in range(len(Filtration)):
        n_active = np.where(np.array(filtration_fun) <= Filtration[p])[0].tolist()
        Active_node = np.unique(n_active)
        if (len(Active_node) == 0):
            betti_0.append(0)
            betti_1.append(0)
        else:
            b = adj[Active_node, :][:, Active_node]
            my_flag = pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2,
                                                   approximation=None)
            x = my_flag["betti"]
            betti_0.append(x[0])
            betti_1.append(x[1])
        n_active.clear()
    return betti_0, betti_1


def topological_embeddings(dataset_Name):
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
    print(data)
    Number_nodes = len(data.y)
    Edge_idx = data.edge_index.numpy()
    Node = range(Number_nodes)
    Edgelist = []
    for i in range(len(Edge_idx[1])):
        Edgelist.append((Edge_idx[0][i], Edge_idx[1][i]))
    # print(Edgelist)
    # a "plain" graph is undirected
    G = nx.DiGraph()

    # give each a node a 'name', which is a letter in this case.
    # G.add_node('a')

    # the add_nodes_from method allows adding nodes from a sequence, in this case a list
    # nodes_to_add = ['b', 'c', 'd']
    G.add_nodes_from(Node)

    # add edge from 'a' to 'b'
    # since this graph is undirected, the order doesn't matter here
    # G.add_edge('a', 'b')

    # just like add_nodes_from, we can add edges from a sequence
    # edges should be specified as 2-tuples
    # edges_to_add = [('a', 'c'), ('b', 'c'), ('c', 'd')]
    G.add_edges_from(Edgelist)

    if dataset_Name == 'cora':
        Node_fil = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 34]
    elif dataset_Name == 'citeseer':
        Node_fil = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 34, 100]
    elif dataset_Name == 'pubmed':
        Node_fil = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 34]
    elif dataset_Name == 'chameleon':
        Node_fil = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50, 100, 200, 400]
    else:
        Node_fil = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100]

    # Node_fil=[2,4,6,8,10,12,14,16,18,20,22,24,30,34] #cora
    # Node_fil=[0,2,4,6,8,10,12,14,16,18,20,22,24,30,34,100]citeseer
    # Node_fil=[1,2,3,4,5,6,7,8,9,10,20,100]#texas, cornell
    # Node_fil=[0,2,4,6,8,10,12,14,16,18,20,22,24,30,34]# pubmed
    topo_betti_0 = []
    topo_betti_1 = []
    Node_Edge = []
    for i in range(Number_nodes):
        print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        subgraph = ego_graph(G, i, radius=2, center=True, undirected=True, distance=None)
        filt = Degree_list(subgraph)
        A_sub = nx.to_numpy_array(subgraph)  # adjacency matrix of subgraph
        fe = Topological_Feature_subLevel(A_sub, filt, Node_fil)
        topo_betti_0.append(fe[0])
        topo_betti_1.append(fe[1])
        Node_Edge.append([subgraph.number_of_nodes(), subgraph.number_of_edges()])
    return topo_betti_0, topo_betti_1

def wise_embeddings_eucledian_mag( Domain_Fe,Label):
    Domain_Fec = pd.DataFrame(Domain_Fe.numpy())
    label = pd.DataFrame(Label.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(label)
    print(Number_nodes)
    fe_len = len(Domain_Fe[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    sel_basis = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    Fec = []
    for i in range(Number_nodes):
        print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        vec = []
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(sel_basis[j])))
        f.clear()
        Fec.append(vec)
    return Fec