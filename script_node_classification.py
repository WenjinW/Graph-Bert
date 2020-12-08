import numpy as np
import torch

from code.NodeClassification import MethodGraphBertNodeClassification
# 放到前面，不然冲突
from code.DatasetLoader import DatasetLoader, Ogbn
from code.MethodBertComp import GraphBertConfig
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


#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 1000 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 150 # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 2000 #2000 # it takes a long epochs to get good results, sometimes can be more than 2000
    elif dataset_name == 'ogbn-arxiv':
        lr = 0.003
        k = 50
        max_epoch = 200
    
    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'raw'
    # --------------------------
    use_pretrain = True
    if use_pretrain:
        print('************ Start Node Classification with Pretrain ************')
    else:
        print('************ Start Node Classification without Pretrain ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------

    data_obj = Ogbn(dataset_name=dataset_name, k=k)
    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    
    pretrained_path = None
    if use_pretrain:
        pretrained_path = './result/PreTrained_GraphBert/' + dataset_name + '/node_reconstruct_model/'
    
    method_obj = MethodGraphBertNodeClassification(bert_config, pretrained_path, dataset_name)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr
    if use_pretrain:
        method_obj.save_pretrained_path = './result/PreTrained_GraphBert/' + dataset_name + '/node_pretrain_classification_model/'
    else:
        method_obj.save_pretrained_path = './result/PreTrained_GraphBert/' + dataset_name + '/node_classification_model/'

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = NewSettings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
