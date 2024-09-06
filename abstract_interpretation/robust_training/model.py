

import pytorch_lightning as pl
from utils import models
from abstract_interpretation.node_deeppoly import NodeDeeppolyVerifier
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from robust_gcn.robust_gcn.robust_gcn import RobustGCNModel
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GCNModel(pl.LightningModule):
    """
    Lightning module for Robust GCN model.
    Train, validate, and test dataset must come from the same graph provided in the concrete domain
    
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.model_folder = config.model_folder
        self.dataset = config.dataset
        self.robust = config.robust
        self.load_weights = config.load_weights
        self.method = config.method
        self.load_model()
        self.lam = 0.5
        
        self.robust_train = config.robust_train
        self.margin_u = config.margin_u
        self.margin = config.margin
        self.criterion = torch.nn.CrossEntropyLoss()
        self.robust_criterion = torch.nn.BCEWithLogitsLoss()
        self.device_name = config.device

    def load_model(self):
        """
        Load the model from the model folder.
        If we use the optimization method, we do not need to load the model now but config
        the model later as it needs the adjacency matrix as input
        """
        if self.robust:
            gcn = models.get_model(f'{self.model_folder}/', 
            self.dataset, is_robust=True, normalize=False)
        else:
            gcn = models.get_model(f'{self.model_folder}/',
             self.dataset, is_robust=False, load_weights=self.load_weights, normalize=False)
        gcn.cuda()
        layers = list(gcn.children())
        self.num_classes = layers[-1].out_channels

        if self.method == 'optim':
            self.dims = [layers[0].in_channels, 32, self.num_classes]
        else:
            self.gcn = gcn
            self.layers = layers

    def configure_optimization_verifier(self, adj, concrete_domain):
        self.concrete_domain = concrete_domain
        self.gcn = RobustGCNModel(adj, self.dims)

    def setup_verifier(self, concrete_domain): 
        self.concrete_domain = concrete_domain
        self.verifier = NodeDeeppolyVerifier(self.layers, concrete_domain, approx='topk', lam=self.lam)
        print(self.verifier.lam)
    
    def forward_optimization(self, x, node_ids: np.array):
        return self.gcn.forward(x, node_ids)

    def forward(self, x, edge_index, edge_weights, batch=None):
        """
        Forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input graph.
        edge_index : torch.Tensor
            Edge indices.
        edge_weights : torch.Tensor
            Edge weights.
        """
        if self.method != 'optim':
            return self.gcn(x, edge_index, edge_weights)
        else:
            nodes = list(range(batch.batch_size))
            node_ids = batch.node_id[nodes].cpu().numpy()
            return self.forward_optimization(self.concrete_domain.x, node_ids)

    def adv_loss(self, batch, diff_lb, coord_x, coord_y):
        """
        Get the perturbation of the worst case for each class, perform the perturbation
        and then calculate the cross entropy loss based on the lower bound of the target
        label and the upper bound of the other labels.
        For this experiment, we use the original network to handle the potential adversarial
        examples
        """
        targets = batch.y[:batch.batch_size]
        max_perturb = diff_lb.argmax(dim=1)

        loss = []
        for i in range(batch.batch_size):
            j = max_perturb[i]
            x = batch.x.detach().clone()
            # get the perturbed input
            x[coord_x[i, j], coord_y[i, j]] = 1 - x[coord_x[i, j], coord_y[i, j]]
            out = self.forward(x, batch.edge_index, batch.edge_attr)
            out = out[i, :]
            loss.append(self.criterion(out.unsqueeze(0), targets[i].unsqueeze(0)))
        
        # return the average loss
        return torch.stack(loss).mean()

    def ce_loss(self, batch, diff_lb, coord_x, coord_y):
        loss = []
        labels = batch.y[:batch.batch_size]
        max_perturb = diff_lb.argmax(dim=1)

        for i in range(batch.batch_size):
            predicted_mask = torch.eye(self.num_classes)[labels[i]].to(self.device_name)
            j = max_perturb[i]
            x = batch.x.detach().clone()
            # get the perturbed input
            # h = self.layers[0](x, batch.edge_index, batch.edge_attr)
            # print('h before', h[0, 23])
            x[coord_x[i, j], coord_y[i, j]] = 1 - x[coord_x[i, j], coord_y[i, j]]
            h = self.layers[0](x, batch.edge_index, batch.edge_attr)
            # print('h after', h[0, 23])
            out = self.verifier.forward(h, batch.edge_index, batch.node_id, batch.edge_attr)
            lb, ub = out._lb[i, :], out._ub[i, :]
            result = lb * predicted_mask + ub * (1 - predicted_mask)
            loss.append(self.criterion(result.unsqueeze(0), labels[i].unsqueeze(0)))
        return torch.stack(loss).mean()

    def optimization_backward(self, node_ids, labels):
        lb, coords = self.gcn.dual_backward(self.concrete_domain.x, node_ids, self.concrete_domain.l, self.concrete_domain.q,
                              target_classes=labels, initialize_omega=True, return_perturbations=True)

        x, y = torch.tensor(coords[:, :, :, 0], device=self.device), torch.tensor(coords[:, :, :, 1], device=self.device)
        non_target_indices = [[i for i in range(self.num_classes) if i != label] for label in labels]
        non_target_indices = torch.tensor(non_target_indices).to(self.device)
        lb = lb.gather(1, non_target_indices)
        # TODO: select the x and y coordinates

        return lb, (x, y)


    def verifier_backward(self, nodes, labels, batch):
        if self.method != 'optim':
            return self.verifier.get_lb_backward(nodes, labels, self.num_classes, batch, self.device_name)
        else:
            node_ids = batch.node_id[nodes].cpu().numpy()
            return self.optimization_backward(node_ids, labels)

    def get_lower_bound(self, nodes, labels, predicted_labels, batch):
        if 'U' in self.hparams.loss:
            # some unlabeled nodes, combine predicted and true labels
            test_mask = batch.test_mask[nodes]
            train_mask = batch.train_mask[nodes]
            labels = labels * train_mask + predicted_labels * test_mask            
        return self.verifier_backward(nodes, labels, batch)
        

    def training_step(self, batch, batch_idx):
        if batch_idx % self.hparams.accumulate_grad_batches == 0 and self.robust_train and self.method == 'poly':
            # The verifier depends on the weights of the nn, so we initialize it for several steps
            self.concrete_domain.edge_index, edge_weights = gcn_norm(self.concrete_domain.edge_index, num_nodes=self.concrete_domain.x.shape[0])
            self.verifier = NodeDeeppolyVerifier(self.layers, self.concrete_domain, approx='max', edge_weights=edge_weights, lam=self.lam)

        labels = batch.y[:batch.batch_size]
        nodes = list(range(batch.batch_size))

        out = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch)
        out = out[nodes, :]

        ce_loss = self.criterion(out, batch.y[nodes])
        self.log('train_loss_ce', ce_loss, batch_size=batch.batch_size)

        diff_lb, (coord_x, coord_y) = self.get_lower_bound(nodes, labels, out.argmax(dim=1), batch)
        robust = (diff_lb > 0).all(dim=1).sum() / batch.batch_size
        
        if self.hparams.loss == 'hinge' or self.hparams.loss == 'hinge-U':
            # each entry of diff_lb should be as large as possible. So we can use hinge loss
            # if the data contains unlabeled nodes, we need to use the other margin
            train_mask = batch.train_mask[nodes]
            test_mask = batch.test_mask[nodes]

            # get different margin based on weather the node is labelled or not
            margin = (self.margin * train_mask + self.margin_u * test_mask).unsqueeze(dim=1)
            robust_loss = F.relu(-diff_lb + margin).mean()
        elif self.hparams.loss == 'bce' or self.hparams.loss == 'bce-U':
            # each entry of diff_lb should be as large as possible. So we can use BCE loss
            self.robust_criterion = torch.nn.BCEWithLogitsLoss(reduce=False)
            train_mask = batch.train_mask[nodes]
            test_mask = batch.test_mask[nodes]

            # get different target based on if the node is labelled or not
            target = torch.ones_like(diff_lb)
            margin = self.margin_u

            robust_loss = self.robust_criterion(diff_lb, target).mean(dim=1)
            hinge_loss = 0.3 * F.relu(-diff_lb + margin).mean(dim=1)
            robust_loss = (robust_loss * train_mask + hinge_loss * test_mask).mean()

        elif self.hparams.loss == 'ce':
            robust_loss = self.ce_loss(batch, diff_lb, coord_x, coord_y)
        elif self.hparams.loss == 'adv':
            robust_loss = self.adv_loss(batch, diff_lb, coord_x, coord_y)
        elif self.hparams.loss == 'pce':
            robust_loss = ce_loss
        else:
            raise NotImplementedError('The loss function is not supported.')

        self.log('train_loss_robust', robust_loss, batch_size=batch.batch_size)
        loss =  robust_loss #+ 0.1 * ce_loss
        self.log('train_loss', loss, batch_size=batch.batch_size)
        self.log('train_robust', robust, batch_size=batch.batch_size)

        self.log('train_acc', (out.argmax(dim=1) == labels).float().mean(), batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.robust_train and self.method == 'poly':
            self.concrete_domain.edge_index, edge_weights = gcn_norm(self.concrete_domain.edge_index, num_nodes=self.concrete_domain.x.shape[0])
            self.verifier = NodeDeeppolyVerifier(self.layers, self.concrete_domain, approx='topk', edge_weights=edge_weights)

        
        # if batch_idx < 2103:
        #     return
        # else:
        #     print(batch.node_id[:batch.batch_size])

        labels = batch.y[:batch.batch_size]
        nodes = list(range(batch.batch_size))

        out = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch)
        out = out[nodes, :]

        ce_loss = self.criterion(out, batch.y[nodes])
        self.log('val_loss_ce', ce_loss, batch_size=batch.batch_size)
        predicted_labels = out.argmax(dim=1)
        diff_lb, (coord_x, coord_y) = self.verifier_backward(nodes, predicted_labels, batch)
        robust = (diff_lb > 0).all(dim=1).sum() / batch.batch_size
        
        if self.hparams.loss == 'hinge' or self.hparams.loss == 'hinge-U':
            robust_loss = F.relu(-diff_lb + self.margin).mean()
        elif self.hparams.loss == 'bce' or self.hparams.loss == 'bce-U':
            # each entry of diff_lb should be as large as possible. So we can use BCE loss
            target = torch.ones_like(diff_lb)
            robust_loss = self.robust_criterion(diff_lb, target)
        elif self.hparams.loss == 'ce':
            robust_loss = self.ce_loss(batch, diff_lb, coord_x, coord_y)
        elif self.hparams.loss == 'adv':
            robust_loss = self.adv_loss(batch, diff_lb, coord_x, coord_y)
        elif self.hparams.loss == 'pce':
            robust_loss = ce_loss
        else:
            raise NotImplementedError('The loss function is not supported.')

        self.log('val_loss_robust', robust_loss, batch_size=batch.batch_size)
        loss = ce_loss + robust_loss
        self.log('val_loss', loss, batch_size=batch.batch_size)
        self.log('val_robust', robust, batch_size=batch.batch_size)

        self.log('val_acc', (out.argmax(dim=1) == labels).float().mean(), batch_size=batch.batch_size)
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0 and self.robust_train:
            # self.verifier = NodeDeeppolyVerifier(self.layers, self.concrete_domain, approx='topk')
            self.verifier = self.get_evaluator_poly()

        labels = batch.y[:batch.batch_size]
        nodes = list(range(batch.batch_size))
        test_nodes = batch.test_mask[nodes]
        num_test = test_nodes.sum()
        train_nodes = batch.train_mask[nodes]
        num_train = train_nodes.sum()

        out = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch)
        out = out[nodes, :]
        predicted_labels = out.argmax(dim=1)
        diff_lb, (_, _) = self.verifier.get_lb_backward(nodes, predicted_labels, self.num_classes, batch, self.device_name)# self.verifier_backward(nodes, predicted_labels, batch)
        
        currect_nodes = out.argmax(dim=1) == labels
        robust_nodes = (diff_lb > 0).all(dim=1)
        # robust = (diff_lb > 0).all(dim=1).sum() / batch.batch_size
        
        if num_test > 0:
            self.log('test_robust_unlabelled', (robust_nodes * test_nodes).sum() / num_test, batch_size=num_test)
            self.log('test_acc_unlabelled', (currect_nodes * test_nodes).sum() / num_test, batch_size=num_test)
        
        if num_train > 0:
            self.log('test_robust_labelled', (robust_nodes * train_nodes).sum() / num_train, batch_size=num_train)
            self.log('test_acc_labelled', (currect_nodes * train_nodes).sum() / num_train, batch_size=num_train)
        self.log('test_robust', robust_nodes.float().mean(), batch_size=batch.batch_size)        
        self.log('test_acc', currect_nodes.float().mean(), batch_size=batch.batch_size)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gcn.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # self.scheduler = CosineAnnealingLR(optimizer, self.hparams.steps)

        return [optimizer] #, [self.scheduler]

    def get_evaluator_poly(self):
        self.concrete_domain.edge_index, edge_weights = gcn_norm(self.concrete_domain.edge_index, num_nodes=self.concrete_domain.x.shape[0])
        if self.method == 'poly':
            return NodeDeeppolyVerifier(self.layers, self.concrete_domain, approx='topk', edge_weights=edge_weights)
        elif self.method == 'optim':
            layers = models.create_GCN_layers(self.gcn.layers)
            return NodeDeeppolyVerifier(layers, self.concrete_domain, approx='topk', edge_weights=edge_weights)
    