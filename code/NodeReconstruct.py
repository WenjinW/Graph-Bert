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


from transformers.modeling_bert import BertPreTrainedModel
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
        self.device = torch.device('cuda:0')

        self.config = config
        # self.bert = MethodGraphBert(config)
        # self.bert = MethodGraphBert(config).to(device=self.device)
        self.bert = torch.nn.Linear(config.x_size, config.hidden_size).to(device=self.device)
        # self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size).to(device=self.device)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids=None, init_pos_ids=None, hop_dis_ids=None, idx=None):

        # outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        # sequence_output = 0
        # for i in range(self.config.k+1):
        #     sequence_output += outputs[0][:,i,:]
        # sequence_output /= float(self.config.k+1)

        # x_hat = self.cls_y(sequence_output)

        outputs = self.bert(raw_features)
        x_hat = self.cls_y(outputs)

        return x_hat


    def train_epoch(self):
        total_loss = 0.0
        total_num = 0
        print("XXX")
        for x in self.dataloader:
            # print("x shape: {}".format(x.shape))
            length = x.shape[0]
            # x_list = np.squeeze(dgl.sampling.random_walk(self.graph, x, length=5)[0])
            x_list = x + np.zeros((x.shape[0], x.shape[1], 6))
            # print("shape x", x_list.shape)
            # print(self.graph.ndata['feat'][x_list].shape)
            x_list_feat = self.graph.ndata['feat'][x_list].to(self.device)
            x_feat = self.graph.ndata['feat'][x].to(self.device)

            
            output = self.forward(x_list_feat)
            loss_train = F.mse_loss(output, x_feat)
            loss_train.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_num += length
                total_loss += loss_train.item()
        return total_loss / total_num

    
    def train_model(self, max_epoch):
        t_begin = time.time()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train()
        self.optimizer.zero_grad()
        self.dataloader = DataLoader(self.data, batch_size=64, shuffle=True, num_workers=4)
        self.ogbn_data = DglNodePropPredDataset('ogbn-arxiv', root='./data')
        self.graph, self.label = self.ogbn_data[0]
        
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            # -------------------------
            loss_train = self.train_epoch()

            self.learning_record_dict[epoch] = {'loss_train': loss_train, 'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):
        print("111")
        self.train_model(self.max_epoch)
        self.bert.save_pretrained(self.save_pretrained_path)
        print("Save pretrained model in {}".format(self.save_pretrained_path))

        return self.learning_record_dict