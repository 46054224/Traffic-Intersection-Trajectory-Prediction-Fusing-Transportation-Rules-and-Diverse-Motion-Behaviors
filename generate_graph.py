from pre_data_graph_sind import *
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sind',
                    help='')
args = parser.parse_args()
dataname = args.dataset

# Data prep
obs_seq_len = 4
pred_seq_len = 16
frame_interval = 1
data_original = './datasets_original/' + dataname + '/'
data_pre = './datasets_graph_' + str(obs_seq_len) + '_' + str(pred_seq_len) + '/' + dataname + '/'

if not os.path.exists(data_pre):
    os.makedirs(data_pre)
    os.makedirs(data_pre + 'train')
    os.makedirs(data_pre + 'val')
    os.makedirs(data_pre + 'test')


################## generate graphs for training ##################
dset_train = TrajectoryDataset(
    data_original + 'train/',
    data_pre + 'train/',
    dataname + '_train.npz',
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, frame_interval=frame_interval)

dset_train = dataset(data_pre + 'train/' + dataname + '_train.npz')
# # loader_train = DataLoader(
# #     dset_train,
# #     batch_size=1,  # This is irrelative to the args batch size parameter
# #     shuffle=True)
#
################## generate graphs for training ##################

################## generate graphs for evaluation ##################
dset_val = TrajectoryDataset(
    data_original + 'val/',
    data_pre + 'val/',
    dataname + '_val.npz',
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, frame_interval=frame_interval)

dset_val = dataset(data_pre + 'val/' + dataname + '_val.npz')
#
# # loader_val = DataLoader(
# #     dset_val,
# #     batch_size=1,  # This is irrelative to the args batch size parameter
# #     shuffle=False)
#
################## generate graphs for evaluation ##################

################## generate graphs for testing ##################
dset_test = TrajectoryDataset(
    data_original + 'test/',
    data_pre + 'test/',
    dataname + '_test.npz',
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, frame_interval=frame_interval)

dset_test = dataset(data_pre + 'test/' + dataname + '_test.npz')
#
# # loader_test = DataLoader(
# #     dset_test,
# #     batch_size=1,  # This is irrelative to the args batch size parameter
# #     shuffle=False)
################## generate graphs for testing ##################
