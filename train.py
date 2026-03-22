from datetime import datetime
from pre_data_graph_sind import *
from metrics import *
import pickle
import argparse
from torch.utils.data import DataLoader
from model import *

parser = argparse.ArgumentParser()

parser.add_argument('--filename', default='8_12_model13_0')
parser.add_argument('--load_interrupted_model', default=True)
parser.add_argument('--cuda', default='2', help='indexes of gpus')

# Data specifc paremeters
parser.add_argument('--low_bound', type=float, default=1.0)
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='sind',
                    help='')

# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_local', type=float, default=0.000001,
                    help='learning rate')
parser.add_argument('--optim_local', default=True)
parser.add_argument('--use_ELR', default=False)
parser.add_argument('--lr_interval', type=int, default=10,
                    help='learning rate changed interval')
parser.add_argument('--fixed_interval', type=bool, default=True,
                    help='learning reduced rate')
parser.add_argument('--lr_reduced', type=float, default=0.9,
                    help='learning reduced rate')
parser.add_argument('--optimizer', default='Adam',
                    help='SGD,Adam')

# Model specific parameters
parser.add_argument('--input_size', type=int, default=21)
parser.add_argument('--output_size', type=int, default=6)
parser.add_argument('--n_spa', type=int, default=3, help='Number of layers')
parser.add_argument('--n_gcn', type=int, default=1, help='Number of layers')
parser.add_argument('--n_cnn', type=int, default=10, help='Number of layers')

stag_epoch = 0

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

index = args.filename

fp = open("summary\\" + args.dataset + index + ".txt", "w+", encoding="utf-8")
fp.write(index + args.dataset)
fp.close()

print('*' * 30)
print("Training initiating....")
print(args)


def graph_loss(V_pred, V_target, low_bound):
    # getLoss
    return student_loss(V_pred, V_target, low_bound)


# Data prep
obs_seq_len = args.obs_seq_len  # 8
pred_seq_len = args.pred_seq_len  # 12
data_original = './datasets_original/' + args.dataset + '/'
data_pre = './datasets_graph_' + str(obs_seq_len) + '_' + str(pred_seq_len) + '/' + args.dataset + '/'

dset_train = dataset(data_pre + 'train/' + args.dataset + '_train.npz')

loader_train = DataLoader(
    dset_train,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=True)

dset_val = dataset(data_pre + 'val/' + args.dataset + '_val.npz')

loader_val = DataLoader(
    dset_val,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=False)

CEloss = torch.nn.CrossEntropyLoss()


def train(epoch):
    global metrics, constant_metrics, loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    c_, c0, c1, c2, c3 = 0, 0, 0, 0, 0

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.type(torch.FloatTensor).cuda() for tensor in batch]
        obs_traj, V1_obs, A1_obs, V1_tr, _, _, vio, prob, act_traj, gt_actions, gt_actions1 = batch

        V1_obs_tmp = V1_obs.permute(0, 3, 1, 2)
        act_traj = act_traj.permute(0, 3, 1, 2, 4)
        type_ = V1_obs_tmp[:, -1, 0, :]

        V_pred, pi, ppi = model.forward(V1_obs_tmp, A1_obs.squeeze(), act_traj)

        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V1_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr, args.low_bound)
            gt_actions = gt_actions1[(type_ != 0).unsqueeze(-1).repeat(1, 1, 4)].reshape(1, -1, 4)
            pi = pi[(type_ != 0).unsqueeze(-1).repeat(1, 1, 4)].reshape(1, -1, 4)

            mi = torch.min(pi, -1)[0].unsqueeze(-1).repeat(1, 1, 4)
            ma = torch.max(pi, -1)[0].unsqueeze(-1).repeat(1, 1, 4)
            pi = (pi - mi) / (ma - mi)
            pi = pi / torch.sum(pi, -1).unsqueeze(-1).repeat(1, 1, 4)

            # l += CEloss(pi, gt_actions)
            # print(torch.argmax(pi, -1))

            l += -torch.mean(pi[gt_actions == 1])
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            optimizer.zero_grad()
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()
            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            if batch_count == len(loader_train) or batch_count % int(len(loader_train) / 10) < args.batch_size:
                print('Train ' + str(int(batch_count / len(loader_train) * 100)) + '% :\t', args.dataset, '\t Epoch:',
                      epoch, '\t Loss:', loss_batch / batch_count)

    if args.use_ELR:
        optimizer.param_groups[keys.index('pi')]['lr'] *= args.lr_reduced  # ExponentialLR
    metrics['train_loss'].append(loss_batch / batch_count)


def vald(epoch):
    global metrics, loader_vals, constant_metrics, stag_epoch
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.type(torch.FloatTensor).cuda() for tensor in batch]
        obs_traj, V1_obs, A1_obs, V1_tr, _, _, vio, prob, act_traj, gt_actions, gt_actions1 = batch

        V1_obs_tmp = V1_obs.permute(0, 3, 1, 2)
        act_traj = act_traj.permute(0, 3, 1, 2, 4)

        V_pred, pi, ppi = model.forward(V1_obs_tmp, A1_obs.squeeze(), act_traj)

        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V1_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr, args.low_bound)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            if batch_count == len(loader_val) or batch_count % int(len(loader_val) / 10) < args.batch_size:
                print('Val ' + str(int(batch_count / len(loader_val) * 100)) + '% :\t', args.dataset, '\t Epoch:',
                      epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'metrics': metrics, 'constant_metrics': constant_metrics}
        torch.save(state, checkpoint_dir + 'val_best.pth')  # OK

        print('Best Result Recorded')
        stag_epoch = 0
    elif args.fixed_interval == False:
        if stag_epoch >= args.lr_interval:
            stag_epoch = 0
            printed = False
            if args.use_ELR == False:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_reduced
            else:
                keys = list(model._modules.keys())
                for index_pg in range(len(optimizer.param_groups)):
                    if index_pg != keys.index('pi'):
                        optimizer.param_groups[index_pg]['lr'] *= args.lr_reduced
                        if printed == False:
                            printed = True
                            print("Learning Rate: ", optimizer.param_groups[index_pg]['lr'])
        else:
            stag_epoch = stag_epoch + 1


# Training settings

print('Data and model loaded')

# Training
checkpoint_dir = './checkpoint/DDGC' + index + '/' + args.dataset + '/'

if os.path.exists(checkpoint_dir + 'val_best.pth') and args.load_interrupted_model:
    with open(checkpoint_dir + 'args.pkl', 'rb') as old_args:
        old_args = pickle.load(old_args)
    old_args.filename = args.filename
    args = old_args
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
    model = ddgc(n_spa=args.n_spa, n_cnn=args.n_cnn, n_gcn=args.n_gcn,
                 input=args.input_size, output=args.output_size, obs_len=args.obs_seq_len,
                 pred_len=args.pred_seq_len).cuda()
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num of Weights:', total_num)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'Adam':
        if args.optim_local == False:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        else:
            keys = list(model._modules.keys())
            params = []
            for key in keys:
                d = {}
                d['params'] = model._modules.get(key).parameters()
                if key == 'pi':
                    d['lr'] = args.lr_local
                params.append(d)
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.01)

    checkpoint = torch.load(checkpoint_dir + 'val_best.pth')

    epoch = checkpoint['epoch'] + 1
    metrics = checkpoint['metrics']
    constant_metrics = checkpoint['constant_metrics']

    model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    keys = list(model._modules.keys())
    for index_pg in range(len(optimizer.param_groups)):
        if index_pg == 0 or index_pg == keys.index('pi'):
            print(keys[index_pg], optimizer.param_groups[index_pg]['lr'])

    print(constant_metrics)
else:
    epoch = 0
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
    print('Checkpoint dir:', checkpoint_dir)
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    model = ddgc(n_spa=args.n_spa, n_cnn=args.n_cnn, n_gcn=args.n_gcn,
                 input=args.input_size, output=args.output_size, obs_len=args.obs_seq_len,
                 pred_len=args.pred_seq_len).cuda()
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num of Weights:', total_num)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'Adam':
        if args.optim_local == False:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        else:
            keys = list(model._modules.keys())
            params = []
            for key in keys:
                d = {}
                d['params'] = model._modules.get(key).parameters()
                if key == 'pi':
                    d['lr'] = args.lr_local
                    print(key, d['lr'])
                params.append(d)
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.01)
    print('New model constructed')

print('Training started ...')
while epoch < args.num_epochs:

    if args.fixed_interval:
        if epoch % args.lr_interval == 0 and epoch != 0:
            if args.use_ELR == False:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_reduced
            else:
                for index_pg in range(len(optimizer.param_groups)):
                    if index_pg != keys.index('pi'):
                        optimizer.param_groups[index_pg]['lr'] *= args.lr_reduced

    starttime = datetime.now()
    train(epoch)
    print('running time: ', str(datetime.now() - starttime))

    starttime = datetime.now()
    vald(epoch)
    print('running time: ', str(datetime.now() - starttime))

    print('*' * 30)
    for k, v in metrics.items():
        if len(v) > 0:
            print(k, v[-1])
    print(constant_metrics)
    print('*' * 30)
    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)
    epoch = epoch + 1
