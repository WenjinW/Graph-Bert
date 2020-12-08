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


from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeConstruct, self).__init__(config)
        self.place = torch.device('cuda:0')

        self.config = config
        self.bert = MethodGraphBert(config).to(device=self.place)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size).to(device=self.place)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids=None, init_pos_ids=None, hop_dis_ids=None, idx=None):
        
        # raw features (N, L, D)
        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = self.cls_y(sequence_output) # (N, D)
        x_hat = torch.unsqueeze(x_hat, dim=1) # (N, L, D)

        scores = torch.sum(x_hat * raw_features, axis=-1) # (N, L)

        return scores

    def train_epoch(self):
        self.train()
        total_loss = 0.0
        total_num = 0
        for x, x_context, x_wl, y in self.dataloader:
            length = x.shape[0]
            
            x_feat = x.to(self.place)
            x_context_feat = x_context.to(self.place)
            x_wl = x_wl.to(self.place)
            
            output = self.forward(x_context_feat, x_wl)
            _, y_pred = torch.max(output, dim=-1)
            y_true = torch.zeros_like(y_pred, dtype=torch.int64) # (N, L)

            loss_train = F.cross_entropy(output, y_true, reduce='sum')
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_num += length
                total_loss += loss_train.item()
        
        return total_loss / total_num

    def train_model(self, max_epoch):
        t_begin = time.time()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dataloader = DataLoader(self.data, batch_size=512, shuffle=True, num_workers=2)
        
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            # -------------------------
            loss_train = self.train_epoch()

            self.learning_record_dict[epoch] = {'loss_train': loss_train, 'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):
        self.train_model(self.max_epoch)
        self.bert.save_pretrained(self.save_pretrained_path)
        print("Save pretrained model in {}".format(self.save_pretrained_path))

        return self.learning_record_dict