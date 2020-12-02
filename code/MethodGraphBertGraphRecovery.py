import torch
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertGraphRecovery(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, pretrained_path):
        super(MethodGraphBertGraphRecovery, self).__init__(config)
        self.device = torch.device('cuda:0')
        self.config = config
        self.bert = MethodGraphBert(config).to(self.device)
        if pretrained_path is not None:
            print("Load pretraiend model from {}".format(pretrained_path))
            self.bert.from_pretrained(pretrained_path).to(self.device)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = sequence_output
        x_norm = torch.norm(x_hat, p=2, dim=1)
        nume = torch.mm(x_hat, x_hat.t())
        deno = torch.ger(x_norm, x_norm)
        cosine_similarity = nume / deno
        return cosine_similarity


    def train_model(self, max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for name in ['raw_embeddings', 'wl_embedding','int_embeddings','hop_embeddings', 'A']:
            self.data[name] = self.data[name].to(self.device)
        
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()

            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'])
            row_num, col_num = output.size()
            loss_train = torch.sum((output - self.data['A'].to_dense()) ** 2)/(row_num*col_num)

            loss_train.backward()
            optimizer.step()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 50 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):

        self.train_model(self.max_epoch)
        self.bert.save_pretrained(self.save_pretrained_path)
        print("Save pretrained model in {}".format(self.save_pretrained_path))

        return self.learning_record_dict