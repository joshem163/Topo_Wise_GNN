from data_loader import *  # Custom data loader script to import datasets
import numpy as np
import torch
import argparse
from logger import *  # Custom logger to track model performance across runs
from wise_emb import *  # Wise embeddings import (custom module)
from models import *  # Model architectures (GCN, SAGE, GAT, MLP) import
from torch_geometric.nn import LINKX
import torch_geometric.transforms as T  # For transforming the graph data

def main():
    # Arguments dictionary specifying model type, dataset, and training configurations
    args = {'model_type': 'LINKX',  # Model architecture: GCN, SAGE, LINKX or GAT
            'dataset': 'texas',  # Dataset: cora, citeseer, pubmed, etc.
            'public_split': 'no',  # Public split: yes for standard datasets (cora, citeseer, pubmed), no for others
            'num_layers': 2,  # Number of GNN layers
            'heads': 1,  # Number of attention heads (for GAT model)
            'batch_size': 32,  # Batch size for training
            'hidden_channels': 32,  # Hidden layer size for the GNN model
            'dropout': 0.5,  # Dropout rate for regularization
            'epochs': 200,  # Number of training epochs
            'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0,  # Optimizer settings
            'runs': 10,  # Number of different random runs for evaluation
            'log_steps': 10,  # Interval of epochs for logging
            'weight_decay': 5e-4, 'lr': 0.01,  # Weight decay for L2 regularization and learning rate
            'hidden_channels_mlp': 20, 'dropout_mlp': 0.2, 'num_layers_mlp': 3}  # Parameters for MLP model

    # Convert args to an object-like structure for easier access
    args = objectview(args)
    print(args)

    # Load dataset-wise embeddings using Wise embeddings based on the dataset choice
    if args.dataset == 'pubmed':
        wise_pca = ContextualPubmed('pubmed')  # Load contextual embeddings for PubMed dataset
        wise = wise_embeddings_eucledian('pubmed')  # Euclidean embeddings for PubMed
        wise_fe1 = torch.tensor(wise)  # Convert embedding to PyTorch tensor
        wise_fe2 = torch.tensor(wise_pca)
        # Concatenate both sets of embeddings
        CC_domain = torch.cat((wise_fe1, wise_fe2), 1).float()
    else:
        wise = wise_embeddings(args.dataset)  # Load wise embeddings for other datasets
        Inc_fe = torch.tensor(wise[0])
        sel_fe = torch.tensor(wise[1])
        CC_domain = torch.cat((Inc_fe, sel_fe), 1).float()

    # Load topological embeddings (Betti numbers) for the dataset
    topo = topological_embeddings(args.dataset)
    topo_betti0 = torch.tensor(topo[0]).float()
    topo_betti1 = torch.tensor(topo[1]).float()
    topo_fe = torch.cat((topo_betti0, topo_betti1), 1)  # Concatenate topological features

    # Load the dataset and apply transformations (e.g., undirected edges, sparse tensor format)
    dataset = load_data(args.dataset, T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    data = dataset[0]  # First graph data object

    # Update the node features with wise embeddings and topo features
    data.x = CC_domain  # Replace original features with wise embeddings
    data.topo = topo_fe  # Add topological features

    # Get topo features and labels
    X = data.topo
    y_true = data.y

    # Initialize the chosen model based on user arguments
    if args.model_type == 'GCN':
        model = GCN(data.num_features, args.hidden_channels, 10, args.num_layers, args.dropout)
    elif args.model_type == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels, 10, args.num_layers, args.dropout)
    elif args.model_type == 'GAT':
        model = GAT(data.num_features, args.hidden_channels, 10, args.num_layers, args.dropout)
    elif args.model_type == 'LINKX':
        model = LINKX(len(data.y), data.num_features, args.hidden_channels, 10, args.num_layers, 1, 1,
                      args.dropout)
    else:
        print('Model does not exist')

    # Initialize MLP models with their respective parameters
    mlp_model = MLP(X.size(-1), args.hidden_channels_mlp, 5, args.num_layers_mlp, args.dropout_mlp)
    mlp_2 = MLP2(15, 100, dataset.num_classes, 3, 0.2)

    # Logger to store and output results
    logger = Logger(args.runs, args)

    # Loop over different runs (cross-validation-like approach)
    for run in range(args.runs):
        # Split data into training, validation, and test based on public split or custom split
        if args.public_split == 'yes':
            train_idx = np.where(data.train_mask)[0]
            valid_idx = np.where(data.val_mask)[0]
            test_idx = np.where(data.test_mask)[0]
        else:
            idx_train = [data.train_mask[i][run] for i in range(len(data.y))]
            train_idx = np.where(idx_train)[0]
            idx_val = [data.val_mask[i][run] for i in range(len(data.y))]
            valid_idx = np.where(idx_val)[0]
            idx_test = [data.test_mask[i][run] for i in range(len(data.y))]
            test_idx = np.where(idx_test)[0]

        # Reset parameters of the models to ensure fresh training
        model.reset_parameters()
        mlp_model.reset_parameters_mlp()
        mlp_2.reset_parameters_mlp2()

        # Optimizers for each model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
        optimizer_mlp2 = torch.optim.Adam(mlp_2.parameters(), lr=0.01)

        # Training loop for each epoch
        for epoch in range(1, 1 + args.epochs):
            # Train the model on the current run's data
            loss = train(model, mlp_model, mlp_2, data, train_idx, optimizer, optimizer_mlp, optimizer_mlp2)
            # Test the model on validation and test sets
            result = test(model, mlp_model, mlp_2, data, train_idx, valid_idx, test_idx)
            logger.add_result(run, result)  # Log the results

            # Log results every `log_steps` epochs
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')

        # Print run statistics
        logger.print_statistics(run)

    # Print overall statistics after all runs
    logger.print_statistics()

if __name__ == "__main__":
    main()
