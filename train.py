from data_loader import *
import numpy as np
import torch
import argparse
from logger import *
from wise_emb import *
from models import *

import torch_geometric.transforms as T

def main():
    args = {'model_type': 'GCN', 'dataset': 'pubmed', 'num_layers': 2, 'heads': 1,
            'batch_size': 32, 'hidden_channels': 32, 'dropout': 0.5, 'epochs': 150,
            'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'runs': 10, 'log_steps': 1,
            'weight_decay': 5e-4, 'lr': 0.01, 'hidden_channels_mlp': 20, 'dropout_mlp': 0.2, 'num_layers_mlp': 3}

    args = objectview(args)
    # print(args)
    # call the dataset here with x,y,train_mask,test_mask,Val_mask, and Adj
    # To add extra feature we can simply update data.x=new fev tensor or we can add new feature
    # dataset = Planetoid(root='/tmp/cora', name='Cora',transform=T.ToSparseTensor())
    # data = dataset[0]

    if args.dataset == 'pubmed':
        wise_pca=ContextualPubmed('pubmed')
        wise = wise_embeddings_eucledian('pubmed')
        wise_fe1 = torch.tensor(wise)
        wise_fe2 = torch.tensor(wise_pca)
        CC_domain = torch.cat((wise_fe1, wise_fe2), 1).float()
    else:
        wise = wise_embeddings(args.dataset)
        Inc_fe = torch.tensor(wise[0])
        sel_fe = torch.tensor(wise[1])
        CC_domain = torch.cat((Inc_fe, sel_fe), 1).float()

    topo = topological_embeddings(args.dataset)
    topo_betti0 = torch.tensor(topo[0]).float()
    topo_betti1 = torch.tensor(topo[1]).float()
    topo_fe = torch.cat((topo_betti0, topo_betti1), 1)

    # To add extra feature we can simply update data.x=new fev tensor or we can add new feature
    dataset = load_data(args.dataset, T.Compose([T.ToUndirected(),T.ToSparseTensor()]))
    data = dataset[0]
    data.x = CC_domain
    data.topo = topo_fe

    X = data.topo
    y_true = data.y
    #data.adj_t = data.adj_t.to_symmetric()

    #     train_idx = np.where(data.train_mask)[0]
    #     valid_idx = np.where(data.val_mask)[0]
    #     test_idx = np.where(data.test_mask)[0]

    model = SAGE(data.num_features, args.hidden_channels, 10, args.num_layers, args.dropout)
    mlp_model = MLP(X.size(-1), args.hidden_channels_mlp, 5, args.num_layers_mlp, args.dropout_mlp)
    # print(mlp_model.parameters())
    mlp_2 = MLP2(15, 100, dataset.num_classes, 3, 0.2)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        idx_train = [data.train_mask[i][run] for i in range(len(data.y))]
        train_idx = np.where(idx_train)[0]
        idx_val = [data.val_mask[i][run] for i in range(len(data.y))]
        valid_idx = np.where(idx_val)[0]
        idx_test = [data.test_mask[i][run] for i in range(len(data.y))]
        test_idx = np.where(idx_test)[0]

        model.reset_parameters()
        mlp_model.reset_parameters_mlp()
        mlp_2.reset_parameters_mlp2()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
        optimizer_mlp2 = torch.optim.Adam(mlp_2.parameters(), lr=0.01)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, mlp_model, mlp_2, data, train_idx, optimizer, optimizer_mlp, optimizer_mlp2)
            result = test(model, mlp_model, mlp_2, data, train_idx, valid_idx, test_idx)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                # print(f'Run: {run + 1:02d}, 'f'Epoch: {epoch:02d}, 'f'Loss: {loss:.4f}, 'f'Train: {100 * train_acc:.2f}%, '
                #      f'Valid: {100 * valid_acc:.2f}% '
                #     f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()