# WISE-GNN: ENHANCING GNNS WITH WISE EMBEDDING AND TOPOLOGICAL ENCODING
Welcome to the repository for Wise-GNN, an innovative machine learning model for Graph representation learning framework designed to enhance graph representation learning, particularly in node classification tasks. Wise-GNN utilizes a  novel approach named "Wise Embeddings", effectively integrating local topological information and positional information in the doamin feature spaces,to improve the performance of existing GNN models. This implementation uses seven benchmark datasets: Cora, Citeseer, PubMed, Texas, Cornell, Wisconsin, and Chameleon. The code is written in Python and utilizes PyTorch and PyTorch Geometric.

![FrameWork-1](https://github.com/joshem163/WISE-GNN/assets/133717791/89269231-6105-4529-bdb1-9cbc59695eb3)

# Model Architecture
The Wise-GNN architecture is built around a core GNN model (GraphSAGE, GCN, GAT, LINKX, or H2GCN) augmented with novel wise embeddings and topological encodings. The model pipeline involves:
- Define a *classlandmarks* for each class in the attribute space.
- Extracting *Wise embeddings* for each node by calculating the distance from the *Classlandmarks*.
- Use *Wise embeddings* for the input of baseline GNN models.
- Incorporating topological embeddings via MLP layers.
- Concatenating the embeddings for final classification.
The combination of *Wise embeddings* and topological features allows the model to learn more expressive node representations.


# Requirements
Wise-GNN depends on the followings:
Pytorch, Pyflagser, networkx 3.1, sklearn 1.3.0, torch_geometric 2.4.0

   
The code is implemented in python 3.11.4. 
# Datasets
In this study,  7 benchmark datasets have been utilized, comprising three homophilic, and four heterophilic datasets, allowing the model to be evaluated across different types of graph structures. The link to access these datasets is provided below:

[CORA](https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip), [CITESEER](https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip), [PUBMED](https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip), [WebKB](https://github.com/bingzhewei/geom-gcn/tree/master/new_data),[Wiki](https://github.com/benedekrozemberczki/MUSAE/tree/master/input)



# Runing the  Experiments
To repeat the experiment a specific dataset, run the train_*.py file with the following command:
- --dataset: Dataset name (options: cora, citeseer, pubmed, texas, cornell, wisconsin, chameleon)
-  --model_type: Baseline Model (GCN, GSAGE, GAT, LINKX, H2GCN), and --public_split: yes/no (yes: cora, citeseer, and pubmed)   

# Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed changes.
- Feel free to open issues for discussion or questions about the code.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

