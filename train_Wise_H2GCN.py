# Import necessary modules and functions
from models import H2GCN, MLP, MLP2, objectview  # Import custom models and utilities
from logger import *  # For logging results of the experiments
from data_loader import *  # Utility for loading datasets
from wise_emb import *  # WISE embeddings (topological feature embeddings)

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize  # Normalization function for embedding


# Training function that trains the models and computes the loss
def train(model, mlp_model, mlp_2, data, train_idx, optimizer, optimizer_mlp, optimizer_mlp2):
    model.train()  # Set GNN model to training mode
    mlp_model.train()  # Set MLP model to training mode
    mlp_2.train()  # Set the second MLP model to training mode
    optimizer.zero_grad()  # Zero the GNN model gradients
    optimizer_mlp.zero_grad()  # Zero the MLP model gradients
    optimizer_mlp2.zero_grad()  # Zero the MLP2 model gradients

    gcn_embedding = model(data)  # Obtain node embeddings from GNN
    mlp_embedding = mlp_model(data.topo)  # Get topological embeddings using MLP

    # Combine GCN embeddings with MLP embeddings
    combined_embedding = torch.cat((gcn_embedding, mlp_embedding), dim=1)

    # MLP2 takes the combined embeddings and outputs predictions
    output = mlp_2(combined_embedding)

    # Compute the loss using negative log likelihood (NLL)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Backpropagate gradients

    # Update weights for all models
    optimizer_mlp2.step()  # Update MLP2 model
    optimizer.step()  # Update GNN model
    optimizer_mlp.step()  # Update MLP model

    return loss.item()  # Return the loss value


# Accuracy function to compute how many predictions match the true labels
def ACC(Prediction, Label):
    correct = Prediction.view(-1).eq(Label).sum().item()  # Count correct predictions
    total = len(Label)  # Total number of labels
    return correct / total  # Return the accuracy ratio


# Testing function to evaluate the models
@torch.no_grad()  # Disable gradient calculation during evaluation
def test(model, mlp_model, mlp_2, data):
    model.eval()  # Set GNN model to evaluation mode
    mlp_model.eval()  # Set MLP model to evaluation mode
    mlp_2.eval()  # Set MLP2 model to evaluation mode

    gcn_out = model(data)  # Get GNN output (node embeddings)
    mlp_out = mlp_model(data.topo)  # Get MLP output (topological embeddings)

    # Combine GCN and MLP embeddings
    combined_out = torch.cat((gcn_out, mlp_out), dim=1)

    # Get final predictions from MLP2
    out = mlp_2(combined_out)
    logits, accs = out, []
    masks = ['train_mask', 'val_mask', 'test_mask']

    # Compute accuracy for train, validation, and test splits
    for mask in masks:
        mask = getattr(data, mask)
        pred = logits[mask].max(1)[1]  # Get predicted class
        acc = ACC(pred, data.y[mask])  # Compute accuracy
        accs.append(acc)

    return accs  # Return list of accuracies for each split


# Main function to run the experiment
def main():
    # Define experiment arguments (these can be passed via argparse as well)
    args = {'model_type': 'H2GCN',  # Model type, here H2GCN is used
            'dataset': 'cora',  # Dataset selection (e.g., texas, cora, citeseer, etc.)
            'public_split': 'yes',  # Public split option for datasets
            'num_layers': 2,  # Number of layers in the GNN model
            'heads': 1,  # Attention heads (relevant if GAT model is used)
            'batch_size': 32,  # Batch size for training
            'hidden_channels': 32,  # Hidden layer size in the GNN model
            'dropout': 0.5,  # Dropout rate to avoid overfitting
            'epochs': 200,  # Number of training epochs
            'opt': 'adam',  # Optimizer type
            'runs': 10,  # Number of random runs for evaluation
            'log_steps': 10,  # Log every 10 epochs
            'weight_decay': 5e-4,  # Weight decay for regularization
            'lr': 0.01,  # Learning rate
            'hidden_channels_mlp': 20,  # Hidden layer size for MLP model
            'dropout_mlp': 0.2,  # Dropout rate for MLP
            'num_layers_mlp': 3}  # Number of layers in MLP model

    args = objectview(args)  # Convert arguments to an object for easier access

    # Load the dataset
    dataset = load_data(args.dataset, None)
    data = dataset[0]

    # Handle custom train/val/test splits for datasets without public split
    if args.public_split == 'no':
        data_train_mask, data_val_mask, data_test_mask = [], [], []
        for run in range(10):
            train_mask = torch.tensor([data.train_mask[i][run] for i in range(len(data.y))])
            data_train_mask.append(train_mask)
            val_mask = torch.tensor([data.val_mask[i][run] for i in range(len(data.y))])
            data_val_mask.append(val_mask)
            test_mask = torch.tensor([data.test_mask[i][run] for i in range(len(data.y))])
            data_test_mask.append(test_mask)

    # Load WISE embeddings based on the dataset
    if args.dataset == 'pubmed':
        wise = wise_embeddings_eucledian('pubmed')
        CC_domain = torch.cat((data.x, torch.tensor(wise)), 1).float()
    else:
        wise = wise_embeddings(args.dataset)
        CC_domain = torch.cat((torch.tensor(wise[0]), torch.tensor(wise[1])), 1).float()

    # Load topological embeddings (e.g., Betti numbers)
    topo = topological_embeddings(args.dataset)
    topo_betti0 = torch.tensor(topo[0]).float()
    topo_betti1 = torch.tensor(topo[1]).float()
    topo_fe = torch.cat((topo_betti0, topo_betti1), 1)

    # Update the dataset with the combined features
    data.x = CC_domain
    data.topo = topo_fe
    X = data.topo
    y_true = data.y

    # Initialize the models (GNN, MLP, MLP2)
    model = H2GCN(in_channels=data.num_features, hidden_channels=args.hidden_channels,
                  out_channels=10, edge_index=data.edge_index, num_nodes=data.num_nodes,
                  num_layers=args.num_layers, dropout=args.dropout).to(data.x.device)
    mlp_model = MLP(X.size(-1), args.hidden_channels_mlp, 5, args.num_layers_mlp, args.dropout_mlp)
    mlp_2 = MLP2(15, 50, dataset.num_classes, args.num_layers_mlp, 0.3)

    logger = Logger(args.runs, args)  # Logger to store results

    # Perform training and evaluation for multiple runs
    for run in range(args.runs):
        if args.public_split == 'no':
            data.train_mask = data_train_mask[run]
            data.val_mask = data_val_mask[run]
            data.test_mask = data_test_mask[run]

        # Reset parameters for each run
        model.reset_parameters()
        mlp_model.reset_parameters_mlp()
        mlp_2.reset_parameters_mlp2()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
        optimizer_mlp2 = torch.optim.Adam(mlp_2.parameters(), lr=0.01)

        # Train and evaluate the model for a given number of epochs
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, mlp_model, mlp_2, data, data.train_mask, optimizer, optimizer_mlp, optimizer_mlp2)
            train_acc, valid_acc, test_acc = test(model, mlp_model, mlp_2, data)
            logger.add_result(run, [train_acc, valid_acc, test_acc])

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)  # Print statistics for each run
    logger.print_statistics()  # Print final statistics across all runs


if __name__ == "__main__":
    main()