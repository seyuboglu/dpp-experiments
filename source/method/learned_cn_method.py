"""
"""
from __future__ import division
import os
import json
import logging
from shutil import copyfile

import numpy as np
import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix

from method import DPPMethod
from learned_cn.metrics import metrics
from learned_cn.utils import (save_checkpoint, load_checkpoint, 
                              save_dict_to_json, RunningAverage)


class LearnedCN(DPPMethod):
    """ GCN method class
    """
    def __init__(self, experiment_dir, params, adjacency, diseases, protein_to_node):
        super(LearnedCN, self).__init__(params, )

        self.experiment_dir = experiment_dir
        self.adjacency = adjacency
        self.diseases = diseases
        self.protein_to_node = protein_to_node

        self.train_method()
    
    def train_method(self):
        """ Trains the underlying model
        """
        dataloaders = fetch_dataloader(['train', 'dev'], 
                                       self.diseases, 
                                       self.protein_to_node, 
                                       self.params)

        train_dl = dataloaders['train']
        dev_dl = dataloaders['dev']

        model = CNModule(self.params, self.adjacency)

        if self.params.cuda:
            model = model.cuda()
        
        optimizer = Adam(model.parameters(), lr=self.params.learning_rate, 
                         weight_decay=self.params.weight_decay)
        
        logging.info("Starting training for {} epoch(s)".format(self.params.num_epochs))
        train_and_evaluate(
            model,
            train_dl,
            dev_dl,
            optimizer,
            bce_loss,
            metrics,
            self.params,
            self.experiment_dir
        )

    def compute_scores(self, train_pos, val_pos):
        """ Compute the scores predicted by GCN.
        Args: 
            ppi_adj_sparse (sp.coo)
            features_sparse (sp.lil)
            train_pos (np.array)
            val_pos (np.array)
            params (dict)
        """
        # Adjacency: Get sparse representation of ppi_adj

        return scores
    

class CNModule(nn.Module):
    
    def __init__(self, params, adjacency):
        super(CNModule, self).__init__()

        # build degree vector
        D = np.sum(adjacency, axis=1)
        D = np.power(D, -0.5)
        self.D = torch.tensor(D, dtype=torch.double)

        # convert adjacency to sparse matrix
        self.A = torch.tensor(adjacency, dtype=torch.double)
        self.A = torch.mul(torch.mul(self.D, self.A), self.D.view(-1, 1))
        N, _ = self.A.shape

        coo = coo_matrix(self.A)
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.DoubleTensor(coo.data)
        self.A_sparse = torch.sparse.DoubleTensor(i, v, torch.Size(coo.shape))

        self.W = nn.Parameter(Variable(torch.ones(1, N, dtype=torch.double), 
                              requires_grad=True))

    def forward(self, input):
        """
        """
        X = input 
        WA = torch.mul(self.W, self.A)
        AWA = torch.matmul(self.A_sparse, WA)
        X = torch.matmul(X, AWA)
        return X 


class DiseaseDataset(Dataset):
    """
    """
    def __init__(self, diseases, protein_to_node, frac_known=0.9):
        """
        """
        self.n = len(protein_to_node)
        self.examples = [disease.to_node_array(protein_to_node) 
                         for disease 
                         in diseases]
        self.frac_known = frac_known
    
    def __len__(self):
        """ 
        Returns the size of the dataset.
        """
        return len(self.examples)
    
    def __getitem__(self, idx):
        nodes = self.examples[idx]
        np.random.shuffle(nodes)
        split = int(self.frac_known * len(nodes))
        known_nodes = nodes[:split]
        hidden_nodes = nodes[split:]
        X = torch.zeros(self.n, dtype=torch.double)
        X[known_nodes] = 1
        Y = torch.zeros(self.n, dtype=torch.double) 
        Y[hidden_nodes] = 1
        return X, Y


def fetch_dataloader(splits, diseases, protein_to_node, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the database
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    datasets = {} 
    for id, disease in diseases.items():
        datasets.setdefault(disease.split, []).append(disease)

    datasets = {split: DiseaseDataset(diseases, protein_to_node) 
                for split, diseases 
                in datasets.items()}

    for split in splits:
        dataloaders[split] = DataLoader(datasets[split], 
                                        batch_size=params.batch_size, 
                                        shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    return dataloaders


def bce_loss(outputs, labels, params):
    """
    """
    num_pos = 1.0 * labels.data.cpu().sum()
    num_neg = labels.data.cpu().nelement() - num_pos
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg / num_pos)
    return bce_loss(outputs, labels)


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and 
                computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches
                 training data
        metrics: (dict) a dictionary of functions that compute a metric using the output 
                and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    loss_avg = RunningAverage()

    outputs_labels = []
    losses = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
        
            if params.cuda:
                labels_batch = labels_batch.cuda(params.cuda_gpu, async=True)
                if isinstance(data_batch, list):
                    data_batch = tuple(x.cuda(params.cuda_gpu, async=True) 
                                       for x 
                                       in data_batch)
                else:
                    data_batch = data_batch.cuda(params.cuda_gpu, async=True)

            # compute model output and loss
            if params.cuda:
                model.cuda(params.cuda_gpu)
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch, getattr(params, 'loss_params', None))

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                if(type(output_batch) is tuple):
                    # output_batch is first element in tuple 
                    output_batch = output_batch[0]
                outputs_labels.append((output_batch.data.cpu(), labels_batch.data.cpu()))

            # update the average loss
            losses.append(loss.data)
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # log output sample of the final batch
    #logging.info('Train output sample:')
    #logging.info('labels[{}]: {}'.format(rand_idx, labels_batch[rand_idx]))
    #logging.info('output[{}]: {}'.format(rand_idx, output_batch[rand_idx]))

    # compute mean of all metrics in summary
    outputs, labels = zip(*outputs_labels)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    summary = {metric : metrics[metric](outputs, labels)
               for metric in metrics}
    summary['loss'] = np.mean(losses)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in summary.items())
    logging.info("- Train metrics: " + metrics_string)


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    outputs_labels = []
    losses = []

    # compute metrics over the dataset
    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
            # move to GPU if available
            if params.cuda:
                labels_batch = labels_batch.cuda(params.cuda_gpu, async=True)
                if isinstance(data_batch, list):
                    data_batch = tuple(x.cuda(params.cuda_gpu, async=True) for x in data_batch)
                else:
                    data_batch = data_batch.cuda(params.cuda_gpu, async=True)

            output_batch = model(data_batch)
            if params.cuda:
                model.cuda(params.cuda_gpu)

            loss = loss_fn(output_batch, labels_batch, getattr(params, 'loss_params', None))
            losses.append(loss.data)


            # extract data from torch Variable, move to cpu, convert to numpy arrays
            if(type(output_batch) is tuple):
                # output_batch is first element in tuple 
                output_batch = output_batch[0]
            outputs_labels.append((output_batch.data.cpu(), labels_batch.data.cpu()))

            t.update()


    # log output sample of the final batch
    #logging.info('Validation output sample:')
    #logging.info('labels[{}]: {}'.format(rand_idx, labels_batch[rand_idx]))
    #logging.info('output[{}]: {}'.format(rand_idx, output_batch[rand_idx]))

    # compute mean of all metrics in summary
    outputs, labels = zip(*outputs_labels)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    summary = {metric:metrics[metric](outputs, labels)
               for metric in metrics}
    summary['loss'] = np.mean(losses)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in summary.items())
    logging.info("- Evaluate metrics: " + metrics_string)

    return summary

def train_and_evaluate(model, train_dataloader, 
                       val_dataloader, optimizer, 
                       loss_fn, metrics, params, 
                       model_dir, restore_file=None, 
                       scheduler=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_primary_metric = 0.0

    for epoch in range(params.num_epochs):
        if scheduler:
            scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        logging.info("Train")
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        logging.info("Evaluate")
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_primary_metric = val_metrics[params.primary_metric]
        is_best = val_primary_metric >= best_val_primary_metric

        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best " + params.primary_metric)

            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            second_best_json_path = os.path.join(model_dir,     
                                                 "metrics_val_second_best_weights.json")

            if os.path.isfile(best_json_path):
                second_best_val_primary_metric = best_val_primary_metric
                # Copy former best val metrics into second best val metrics
                copyfile(best_json_path, second_best_json_path)

            best_val_primary_metric = val_primary_metric
            # Save best val metrics in a json file in the model directory
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)