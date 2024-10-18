import math
import os
import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from utils import *
from data_reading import *
from models import NEGCDTI
import torch 
import torch.nn as nn
import torch.nn.functional as F
from train_test_split import kf_split
from sklearn.metrics import roc_curve, auc,average_precision_score
import matplotlib.pyplot as plt
import argparse
from config_init import get_config
torch.cuda.manual_seed(1223)
from cluster import Cluster

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__=="__main__":
    config = get_config()
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    drug_sim_file = config.drug_sim_file_topofallfeature
    target_sim_file = config.target_sim_file_topofallfeature
    dti_mat = config.dti_mat_topofallfeature
    device = config.device_topofallfeature
    n_splits = config.n_splits_topofallfeature
    hgcn_dim = config.hgcn_dim_topofallfeature
    dropout = config.dropout_topofallfeature
    epoch_num = config.epoch_num_topofallfeature
    lr = config.lr_topofallfeature
    topk = config.topk_topofallfeature
    epoch_interv = config.epoch_interv_topofallfeature
    lr_decay = config.lr_decay_topofallfeature
    depth_e1 = config.depth_e1_topofallfeature
    depth_e2 = config.depth_e2_topofallfeature
    embed_dim = config.embed_dim_topofallfeature
    drop_ratio = config.drop_ratio_topofallfeature
    in_dim = config.in_dim_topofallfeature
    temperature = config.temperature_topofallfeature
    xs = config.xs_topofallfeature
    thd = config.thd_topofallfeature
    tht = config.tht_topofallfeature
    jsd = config.jsd_topofallfeature
    jst = config.jst_topofallfeature
    kd = config.kd_topofallfeature
    kt = config.kt_topofallfeature
    # data reading
    data_folder = os.path.join(root_path,dataset)
    drug_sim_path = os.path.join(data_folder,drug_sim_file)
    target_sim_path = os.path.join(data_folder,target_sim_file)
    DTI_path = os.path.join(data_folder,dti_mat)
    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    SR = SR[300:]
    SR = SR.flatten()
    SR = string_float(SR)
    SR = SR.reshape(drug_num, drug_num)
    SD = SD[200:]
    SD = SD.flatten()
    SD = string_float(SD)
    SD = SD.reshape(target_num, target_num)
    SR = Global_Normalize(SR)
    SD = Global_Normalize(SD)
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    A_orig_list = A_orig.flatten()
    drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)
    negtive_index_arr = np.where(A_orig_arr==0)[0]
    negative_index = torch.LongTensor(negtive_index_arr)
    drug_gau = get_gaussian(A_orig)
    tar_gau = get_gaussian(A_orig.T)
    np.fill_diagonal(drug_gau, 0)
    np.fill_diagonal(tar_gau, 0)
    cluster_drugs, indices_drugs = Cluster(drug_gau, thd, jsd, kd)
    cluster_targets, indices_targets = Cluster(tar_gau, tht, jst, kt)
    train_smile_name = "Data/" + dataset + "/drugs_smiles_" + str(1) + "_gram.npy"
    train_pro_name = "Data/" + dataset + "/target_" + str(1) + "_gram.npy"
    smiles_feature, pro_feature = data_loader(train_smile_name, train_pro_name)
    smiles_feature = smiles_feature.to(device)
    pro_feature = pro_feature.to(device)
    # kfold CV
    train_all, test_all = kf_split(known_sample,n_splits)
    overall_auroc = 0
    overall_aupr = 0
    overall_f1 = 0
    overall_acc = 0
    overall_recall = 0
    overall_specificity = 0
    overall_precision = 0
    for fold_int in range(n_splits):
        print('fold_int:',fold_int)
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]
        A_train_tensor = torch.LongTensor(A_train)
        A_test_tensor = torch.LongTensor(A_test)
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1
        A_train_mask = A_train_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_test_mask = A_test_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_unknown_mask = 1 - A_orig
        A_train_mat = A_train_mask
        # G is the normalized adjacent matrix
        G = Construct_G(A_train_mat,SR,SD).to(device)
        # sample the negative samples
        train_neg_mask_candidate = get_negative_samples(A_train_mask,drug_dissimmat)
        train_neg_mask = np.multiply(train_neg_mask_candidate, A_unknown_mask)
        train_negative_index = np.where(train_neg_mask.flatten() ==1)[0]
        training_negative_index = torch.tensor(train_negative_index)
        # initizalize the model
        train_W = torch.randn(hgcn_dim, hgcn_dim).to(device)
        train_W = nn.init.xavier_normal_(train_W)
        gcn_model = NEGCDTI(depth_e1=depth_e1,
                                depth_e2=depth_e2,
                                embed_dim=embed_dim,
                                in_dim=in_dim,
                                hgcn_dim=hgcn_dim,
                                train_W = train_W,
                                dropout=dropout,
                                drop_ratio=drop_ratio).to(device)
        gcn_optimizer = torch.optim.Adam(list(gcn_model.parameters()),lr=lr)
        # train procedure
        gcn_model.train()
        for epoch in range(epoch_num):
            #prediction results
            if epoch % 100 == 0 and epoch != 0:
                gcn_optimizer.param_groups[0]['lr'] *= lr_decay
            A_hat, features = gcn_model(smiles_feature, pro_feature,G,drug_num,target_num)
            A_hat_list = A_hat.view(1,-1)
            train_sample = A_hat_list[0][A_train_tensor]
            train_score = torch.sigmoid(train_sample)
            nega_sample = A_hat_list[0][training_negative_index]
            nega_score = torch.sigmoid(nega_sample)
            drugs_feature = features[0:drug_num]
            targets_feature = features[drug_num:]
            drugs_view1 = Cluster_view(drugs_feature, cluster_drugs)
            drugs_view3 = Cluster_core(drugs_feature, cluster_drugs)
            drugs_view2 = drugs_feature[indices_drugs]
            drugs_cl_loss = InfoNCE(drugs_view1, drugs_view2, drugs_view3, temperature, drug_num)
            targets_view1 = Cluster_view(targets_feature, cluster_targets)
            targets_view3 = Cluster_core(targets_feature, cluster_targets)
            targets_view2 = targets_feature[indices_targets]
            targets_cl_loss = InfoNCE(targets_view1, targets_view2, targets_view3, temperature, target_num)
            # calculate the loss
            loss_r = loss_function(train_score,nega_score,drug_num,target_num)
            cl_loss = drugs_cl_loss + targets_cl_loss
            loss = loss_r + xs*cl_loss
            los_ = loss.detach().item()
            los_r = loss_r.detach().item()
            los_cl = cl_loss.detach().item()
            print(f"Epoch: {epoch+1}, Loss: {los_}, Loss_r: {los_r}, Loss_cl: {los_cl}")
            gcn_optimizer.zero_grad()
            loss.backward()
            gcn_optimizer.step()
        # test procedure
        gcn_model.eval()
        test_neg_mask_candidate = get_negative_samples(A_test_mask,drug_dissimmat)
        test_neg_mask = np.multiply(test_neg_mask_candidate, A_unknown_mask)
        test_negative_index = np.where(test_neg_mask.flatten() ==1)[0]
        test_negative_index = torch.tensor(test_negative_index)
        positive_samples = A_hat_list[0][A_test_tensor].detach().cpu().numpy()
        negative_samples = A_hat_list[0][test_negative_index].detach().cpu().numpy()
        positive_labels = np.ones_like(positive_samples)
        negative_labels = np.zeros_like(negative_samples)
        labels = np.hstack((positive_labels,negative_labels))
        scores = np.hstack((positive_samples,negative_samples))
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(labels,scores)
        # print('TP:',TP)
        # print('FP:',FP)
        # print('FN:',FN)
        # print('TN:',FN)
        # print('fpr:',fpr)
        # print('tpr:',tpr)
        print('auroc:',auroc)
        print('aupr:',aupr)
        print('f1_score:',f1_score)
        print('acc:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
        overall_auroc += auroc
        overall_aupr += aupr
        overall_f1 += f1_score
        overall_acc += accuracy
        overall_recall += recall
        overall_specificity +=specificity
        overall_precision += precision
    auroc_ = overall_auroc/n_splits
    aupr_ = overall_aupr/n_splits
    f1_ = overall_f1/n_splits
    acc_ = overall_acc/n_splits
    recall_ = overall_recall/n_splits
    specificity_ = overall_specificity/n_splits
    precision_ = overall_precision/n_splits
    print('mean_auroc:',auroc_)
    print('mean_aupr:',aupr_)
    print('mean_f1:',f1_)
    print('mean_acc:',acc_)
    print('mean_recall:',recall_)
    print('mean_specificity:',specificity_)
    print('mean_precision:',precision_)


