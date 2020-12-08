import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
from dgl.dataloading.pytorch import NodeDataLoader
from dgl.sampling import random_walk
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.nodeproppred import Evaluator


from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, pretrained_path, dataset_name):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.place = torch.device('cuda:0')
        self.config = config
        self.bert = MethodGraphBert(config).to(device=self.place)
        # load from pretrained model if necessary
        if pretrained_path is not None:
            self.bert = self.bert.from_pretrained(pretrained_path).to(device=self.place)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size).to(device=self.place)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size).to(device=self.place)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size).to(device=self.place)
        self.init_weights()

        self.evaluator = Evaluator(dataset_name)

    def forward(self, context_feat, wl_role_ids=None, node_feat=None, init_pos_ids=None, hop_dis_ids=None, idx=None):
        residual_h, residual_y = self.residual_term(node_feat)
        outputs = self.bert(context_feat, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        labels = self.cls_y(sequence_output)
        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)
    
    def residual_term(self, node_feat):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(node_feat), self.res_y(node_feat)
        elif self.config.residual_type == 'graph_raw':

            return torch.spmm(self.data['A'], self.res_h(self.data['X'])), torch.spmm(self.data['A'], self.res_y(self.data['X']))


    def train_epoch(self):
        self.train()
        total_loss, total_right, total_num = 0.0, 0.0, 0.0
        for x, x_context, x_wl, y in self.dataloader:
            length = x.shape[0]
            x_feat = x.to(self.place)
            x_context_feat = x_context.to(self.place)
            x_wl = x_wl.to(self.place)
            y = y.to(self.place)

            output = self.forward(x_context_feat, wl_role_ids=x_wl, node_feat=x_feat)
            loss_train = F.cross_entropy(output, y, reduction='sum')
            
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_num += length
                total_loss += loss_train.item()
                _, pred = torch.max(output, dim=1)
                total_right += torch.sum(pred == y)

        return total_loss / total_num, total_right / total_num
    
    def eval_epoch(self):
        self.eval()
        total_loss, total_right, total_num = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, x_context, x_wl, y in self.dataloader:
                length = x.shape[0]
                x_feat = x.to(self.place)
                x_context_feat = x_context.to(self.place)
                x_wl = x_wl.to(self.place)
                y = y.to(self.place)

                output = self.forward(x_context_feat, wl_role_ids=x_wl, node_feat=x_feat)
                loss_train = F.cross_entropy(output, y, reduction='sum')
                
                total_num += length
                total_loss += loss_train.item()
                _, pred = torch.max(output, dim=1)
                total_right += torch.sum(pred == y)

        return total_loss / total_num, total_right / total_num

    
    def train_model(self, max_epoch):
        t_begin = time.time()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dataloader = DataLoader(self.data, batch_size=512, shuffle=True, num_workers=2)
        # self.val_dataloader = DataLoader(self.data[self.data.split['valid']], batch_size=512, shuffle=False, num_workers=2)

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            loss_train, acc_train = self.train_epoch()
            self.learning_record_dict[epoch] = {'loss_train': loss_train, "acc_train": acc_train, 'time': time.time() - t_epoch_begin}
            if epoch % 1 == 0:
                print('| Epoch: {:04d} |'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train),
                      'acc_train: {:.3f} |'.format(acc_train*100),
                      'time: {:.4f}s |'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")

        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):
        self.train_model(self.max_epoch)
        self.bert.save_pretrained(self.save_pretrained_path)
        print("Save Node Classification model in {}".format(self.save_pretrained_path))

        return self.learning_record_dict