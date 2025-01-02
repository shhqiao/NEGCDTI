import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, nargs='?', default="Data",help="root dataset path")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, nargs='?', default="davis",help="setting the dataset")
    parse.add_argument('-device', '--device_topofallfeature', type=str, nargs='?', default="cuda:0",help="setting the cuda device")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default= 10,help="k fold")
    parse.add_argument('-drug_sim_file', '--drug_sim_file_topofallfeature', type=str, nargs='?', default="drug_affinity_mat.txt",help="setting the drug similarity file")
    parse.add_argument('-target_sim_file', '--target_sim_file_topofallfeature', type=str, nargs='?', default="target_affinity_mat.txt",help="setting the target similarity file")
    parse.add_argument('-dti_mat', '--dti_mat_topofallfeature', type=str, nargs='?', default="dti_mat.xlsx",help="setting the dti matrix file")
    parse.add_argument('-kd', '--kd_topofallfeature', type=float, default=1)
    parse.add_argument('-kt', '--kt_topofallfeature', type=float, default=-0.75)
    parse.add_argument('-jsd', '--jsd_topofallfeature', type=float, default=5)
    parse.add_argument('-jst', '--jst_topofallfeature', type=float, default=1)
    parse.add_argument('-xs', '--xs_topofallfeature', type=float, default=0.05)
    parse.add_argument('-thd', '--thd_topofallfeature', type=float, default=0.55)
    parse.add_argument('-tht', '--tht_topofallfeature', type=float, default=0.7)
    parse.add_argument('-temperature', '--temperature_topofallfeature', type=float, default=0.1)
    parse.add_argument('-drop_ratio', '--drop_ratio_topofallfeature', type=float, default=0.)
    parse.add_argument('-in_dim', '--in_dim_topofallfeature', type=int, nargs='?', default=512)
    parse.add_argument('-hgcn_dim', '--hgcn_dim_topofallfeature', type=int, nargs='?', default=512)
    parse.add_argument('-dropout', '--dropout_topofallfeature', type=float, nargs='?', default=0)
    parse.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, nargs='?', default=1200)
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default=0.00005)
    parse.add_argument('-topk', '--topk_topofallfeature', type=int, nargs='?', default=1)
    parse.add_argument('-epoch_interv', '--epoch_interv_topofallfeature', type=int, nargs='?', default=10)
    parse.add_argument('-depth_e1', '--depth_e1_topofallfeature', type=int, nargs='?', default=1)
    parse.add_argument('-depth_e2', '--depth_e2_topofallfeature', type=int, nargs='?', default=1)
    parse.add_argument('-embed_dim', '--embed_dim_topofallfeature', type=int, nargs='?', default=256)
    parse.add_argument('-lr_decay', '--lr_decay_topofallfeature', type=float, nargs='?', default=1)
    config = parse.parse_args()
    return config
