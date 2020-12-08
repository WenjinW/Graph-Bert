import numpy as np
import torch
import faulthandler
faulthandler.enable()
from ogb.nodeproppred import DglNodePropPredDataset

from code.DatasetLoader import DatasetLoader, Ogbn
from code.MethodBertComp import GraphBertConfig
from code.NodeReconstruct import MethodGraphBertNodeConstruct
from code.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
from code.ResultSaving import ResultSaving
from code.Settings import Settings, NewSettings

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'ogbn-arxiv'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717
elif dataset_name == 'ogbn-arxiv':
    nclass = 40
    nfeature = 128
    ngraph = 1



#---- Pre-Training Task #1: Graph Bert Node Attribute Reconstruction (Cora, Citeseer, and Pubmed) ----
if __name__ == "__main__":
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 200 # ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.001
        k = 7
        max_epoch = 200 # ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 200 # it takes a long epochs to converge, probably more than 2000
    elif dataset_name == 'ogbn-arxiv':
        lr = 0.0001
        k = 50
        max_epoch = 50
    
    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', Pre-training, Node Attribute Reconstruction.')
    # ---- objection initialization setction ---------------

    data_obj = Ogbn(dataset_name=dataset_name, k=k)
    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertNodeConstruct(bert_config)
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr
    method_obj.save_pretrained_path = './result/PreTrained_GraphBert/' + dataset_name + '/node_reconstruct_model/'

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(k) + '_node_reconstruction'

    setting_obj = NewSettings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------
    # method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_reconstruct_model/')
    
    print('************ Finish ************')
#------------------------------------
