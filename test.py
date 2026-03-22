import pickle
import glob
from metrics import *
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.stats import t
import pandas as pd


def test(model, n2=20, bound=1, use_scale=True, order=1, write_csv=False, no_pedestrian=False, csv_filename=None,
         multimodal=False):
    starttime = datetime.now()
    if write_csv:
        meta_columns = ['trackId', 'initialFrame', 'finalFrame', 'Frame_nums', 'width', 'lenght', 'class',
                        'CrossType', 'Signal_Violation_Behavior']
        smooth_columns = ['track_id', 'frame_id', 'agent_type', 'x', 'y']
        light_columns = ['frame_id', '0', '1', '2', '3', '4', '5', '6', '7']
        meta_ = pd.DataFrame(data=None, columns=meta_columns)
        smooth_ = pd.DataFrame(data=None, columns=smooth_columns)
        light_ = pd.DataFrame(data=None, columns=light_columns)

    c0, c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0, 0
    a0, a1, a2, a3, a4, a5, a6 = 0, 0, 0, 0, 0, 0, 0
    f0, f1, f2, f3, f4, f5, f6 = 0, 0, 0, 0, 0, 0, 0
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = -1

    for batch in loader_test:
        step += 1
        if write_csv:
            print(step)
        # Get data
        batch = [tensor.type(torch.FloatTensor).cuda() for tensor in batch]
        obs_traj, V1_obs, A1_obs, V1_tr, type, _, vio, prob, act_traj, gt_action, _ = batch

        if no_pedestrian:
            obs_traj = obs_traj[:, :, :2]
            V2_obs = V1_obs
            V1_obs = V2_obs[V2_obs[:, :, :, -1].bool().unsqueeze(-1).repeat(1, 1, 1, V2_obs.shape[-1])].reshape(
                V2_obs.shape[0], V2_obs.shape[1], -1, V2_obs.shape[-1])
            act_traj = act_traj[V2_obs[:, :, :, -1].bool().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2, 5)].reshape(
                act_traj.shape[0], act_traj.shape[1], -1, act_traj.shape[3], act_traj.shape[4])
            if V1_obs.shape[-2] != 0:
                V1_tr = V1_tr[V2_obs[:, 0:1, :, -1:].bool().repeat(1, V1_tr.shape[1], 1, V1_tr.shape[-1])].reshape(
                    V1_tr.shape[0], V1_tr.shape[1], -1, V1_tr.shape[-1])
                A2_obs = A1_obs
                A1_obs = A1_obs[torch.mul(V2_obs[0, 0, :, -1:].bool().repeat(1, A1_obs.shape[-2]),
                                          V2_obs[0, 0, :, -1:].bool().repeat(1, A1_obs.shape[-2]).T
                                          ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(A1_obs.shape[0],
                                                                                           A1_obs.shape[1],
                                                                                           1, 1, A1_obs.shape[-1])]
                A1_obs = A1_obs.reshape(A2_obs.shape[0], A2_obs.shape[1], -1, A2_obs.shape[-1])
                A1_obs = A1_obs.reshape(A1_obs.shape[0], A1_obs.shape[1], int(A1_obs.shape[2] ** 0.5),
                                        int(A1_obs.shape[2] ** 0.5), A1_obs.shape[-1])
            else:
                continue
        else:
            obs_traj = obs_traj[:, :, :2]
            V2_obs = V1_obs

        V1_obs_tmp = V1_obs.permute(0, 3, 1, 2)
        act_traj = act_traj.permute(0, 3, 1, 2, 4)

        c0 += torch.sum((V2_obs[0, 0, :, -1] == 0).float())
        c1 += torch.sum((V2_obs[0, 0, :, -1] == 1).float())
        c2 += torch.sum((V2_obs[0, 0, :, -1] == 2).float())
        c3 += torch.sum((V2_obs[0, 0, :, -1] == 3).float())
        c4 += torch.sum((V2_obs[0, 0, :, -1] == 4).float())
        c5 += torch.sum((V2_obs[0, 0, :, -1] == 5).float())
        c6 += torch.sum((V2_obs[0, 0, :, -1] == 6).float())
        type_ = V2_obs[0, 0, :, -1]
        if no_pedestrian:
            type_ = type_[type_.bool()]

        V_pred, pi, index = model.forward(V1_obs_tmp, A1_obs.squeeze(), act_traj)

        V1_obs = V1_obs[:, :, :, :2]
        obs_traj = obs_traj[:, :, :2]
        V2_tr = V1_tr
        V1_tr = V1_tr[:, :, :, :2]

        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V1_tr.squeeze(0)
        V_pred = V_pred.squeeze(0)
        num_of_objs = V_pred.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V1_obs.data.cpu().numpy().squeeze(0).copy(),
                                                V_x[0, :, :].copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V1_tr.data.cpu().numpy().squeeze(0).copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        raw_data_dict[step]['type'] = type_.cpu().numpy()
        ade_best = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        if use_scale:
            scaler = torch.from_numpy(
                np.concatenate((np.linspace(1.75, 0.75, num=(int)(V_pred.shape[0] / 2)).repeat(V_pred.shape[1]).reshape(
                    ((int)(V_pred.shape[0] / 2), V_pred.shape[1])),
                                np.linspace(0.75, 1.75, num=V_pred.shape[0] - (int)(V_pred.shape[0] / 2)).repeat(
                                    V_pred.shape[1]).reshape(
                                    (V_pred.shape[0] - (int)(V_pred.shape[0] / 2), V_pred.shape[1]))), axis=0))
            # 1.75 0.75
        else:
            scaler = 1.0

        for k in range(n2):
            mx = torch.tensor(
                t.rvs(df=(torch.exp(V_pred[:, :, 2]).data.cpu() + bound) * scaler,
                      scale=torch.exp(V_pred[:, :, 4]).data.cpu(),
                      loc=(V_pred[:, :, 0]).data.cpu(),
                      size=(V_pred.shape[0], V_pred.shape[1]), random_state=None))
            my = torch.tensor(
                t.rvs(df=(torch.exp(V_pred[:, :, 3]).data.cpu() + bound) * scaler,
                      scale=torch.exp(V_pred[:, :, 5]).data.cpu(),
                      loc=(V_pred[:, :, 1]).data.cpu(),
                      size=(V_pred.shape[0], V_pred.shape[1]), random_state=None))
            V_p = torch.cat((mx.unsqueeze(-1), my.unsqueeze(-1)), -1)
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_p.data.cpu().numpy().copy(),
                                                       V_x[-1, :, :].copy())

            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])

                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)
                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_best.append(raw_data_dict[step]['pred'][np.argmin(np.array(ade_ls[n]))][:, n])
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))
            if type_[n] == 0:
                a0 += ade_bigls[-1]
                f0 += fde_bigls[-1]
            if type_[n] == 1:
                a1 += ade_bigls[-1]
                f1 += fde_bigls[-1]
            if type_[n] == 2:
                a2 += ade_bigls[-1]
                f2 += fde_bigls[-1]
            if type_[n] == 3:
                a3 += ade_bigls[-1]
                f3 += fde_bigls[-1]
            if type_[n] == 4:
                a4 += ade_bigls[-1]
                f4 += fde_bigls[-1]
            if type_[n] == 5:
                a5 += ade_bigls[-1]
                f5 += fde_bigls[-1]
            if type_[n] == 6:
                a6 += ade_bigls[-1]
                f6 += fde_bigls[-1]

        if write_csv:
            if multimodal:
                pred = np.tile(raw_data_dict[step]['pred'], reps=1)
                raw_data_dict[step]['pred'] = np.zeros((pred.shape[0], pred.shape[2], pred.shape[1], pred.shape[3]))
                for pidx in range(pred.shape[2]):
                    raw_data_dict[step]['pred'][:, pidx] = pred[:, :, pidx]
                raw_data_dict[step]['ade_best'] = raw_data_dict[step]['pred'].reshape(-1,
                                                                                      raw_data_dict[step]['pred'].shape[
                                                                                          2],
                                                                                      raw_data_dict[step]['pred'].shape[
                                                                                          3])
            else:
                raw_data_dict[step]['ade_best'] = np.tile(ade_best, reps=1)

            type_name = []
            for agent_idx in range(raw_data_dict[step]['obs'].shape[1]):
                if type_[agent_idx] == 0:
                    agent_type = 'pedestrian'
                if type_[agent_idx] == 1:
                    agent_type = 'motorcycle'
                if type_[agent_idx] == 2:
                    agent_type = 'car'
                if type_[agent_idx] == 3:
                    agent_type = 'bus'
                if type_[agent_idx] == 4:
                    agent_type = 'bicycle'
                if type_[agent_idx] == 5:
                    agent_type = 'truck'
                if type_[agent_idx] == 6:
                    agent_type = 'tricycle'
                type_name.append(agent_type)
            type_name = np.array(type_name)

            tid = np.linspace(start=meta_.shape[0], stop=meta_.shape[0] + raw_data_dict[step]['obs'].shape[1] - 1,
                              num=raw_data_dict[step]['obs'].shape[1])[:, np.newaxis]
            initframe = step * (raw_data_dict[step]['obs'].shape[0] +
                                raw_data_dict[step]['trgt'].shape[0]) * np.ones(tid.shape[0])[:, np.newaxis]
            finalframe = (initframe + raw_data_dict[step]['obs'].shape[0] + raw_data_dict[step]['trgt'].shape[0] - 1)
            fnum = finalframe - initframe + 1
            realtraj = np.concatenate(
                (tid, initframe, finalframe, fnum, np.ones((fnum.shape[0], meta_columns.__len__() - 4))),
                axis=1).astype(np.int)
            meta = pd.DataFrame(data=realtraj, columns=meta_columns)
            meta.loc[:, 'class'] = type_name
            meta_ = pd.concat((meta_, meta), axis=0)

            light = np.concatenate((V2_obs[0, :, 0, -9:-1].cpu().numpy(), V2_tr[0, :, 0, 2:-3].cpu().numpy()), axis=0)
            light = np.concatenate((np.linspace(start=initframe[0, 0], stop=finalframe[0, 0], num=int(fnum[0, 0]))[:,
                                    np.newaxis], light), axis=1)
            light = pd.DataFrame(data=light, columns=light_columns)
            light_ = pd.concat((light_, light), axis=0)

            realtraj = np.concatenate((raw_data_dict[step]['obs'], raw_data_dict[step]['trgt']), axis=0)
            realtid = np.tile(tid, (1, realtraj.shape[0])).reshape(-1)[:, np.newaxis]
            fid = np.tile(
                np.linspace(start=initframe[0, 0], stop=finalframe[0, 0], num=int(fnum[0, 0])), realtraj.shape[1])[:,
                  np.newaxis]
            xy = np.zeros((fid.shape[0], 2))
            for frame_idx in range(realtraj.shape[0]):
                xy[frame_idx:xy.shape[0]:realtraj.shape[0]] = realtraj[frame_idx]

            realtraj = np.concatenate((realtid, fid, np.ones((xy.shape[0], 1)), xy), axis=1)
            smooth = pd.DataFrame(data=realtraj, columns=smooth_columns)
            smooth.loc[:, 'agent_type'] = \
                np.tile(type_name[:, np.newaxis],
                        (1, raw_data_dict[step]['obs'].shape[0] + raw_data_dict[step]['trgt'].shape[0])).reshape(-1)
            smooth_ = pd.concat((smooth_, smooth), axis=0)

            tid = np.linspace(start=meta_.shape[0], stop=meta_.shape[0] + raw_data_dict[step]['ade_best'].shape[0] - 1,
                              num=raw_data_dict[step]['ade_best'].shape[0])[:, np.newaxis]
            initframe += raw_data_dict[step]['obs'].shape[0]
            fnum = finalframe - initframe + 1
            if multimodal:
                type_name = np.tile(type_name[np.newaxis, :], (n2, 1)).reshape(-1)
                initframe = np.tile(initframe, (n2, 1))
                finalframe = np.tile(finalframe, (n2, 1))
                fnum = np.tile(fnum, (n2, 1))
            predtraj = np.concatenate(
                (tid, initframe, finalframe, fnum, np.ones((fnum.shape[0], meta_columns.__len__() - 4))),
                axis=1).astype(np.int)
            meta = pd.DataFrame(data=predtraj, columns=meta_columns)
            meta.loc[:, 'class'] = type_name
            meta_ = pd.concat((meta_, meta), axis=0)

            predtraj = raw_data_dict[step]['ade_best']
            predtid = np.tile(tid, (1, predtraj.shape[1])).reshape(-1)[:, np.newaxis]
            fid = np.tile(
                np.linspace(start=initframe[0, 0], stop=finalframe[0, 0], num=int(fnum[0, 0])), predtraj.shape[0])[:,
                  np.newaxis]
            xy = np.zeros((fid.shape[0], 2))
            for frame_idx in range(predtraj.shape[1]):
                xy[frame_idx:xy.shape[0]:predtraj.shape[1]] = predtraj[:, frame_idx]

            predtraj = np.concatenate((predtid, fid, np.ones((xy.shape[0], 1)), xy), axis=1)
            smooth = pd.DataFrame(data=predtraj, columns=smooth_columns)
            smooth.loc[:, 'agent_type'] = \
                np.tile(type_name[:, np.newaxis],
                        (1, raw_data_dict[step]['ade_best'].shape[1])).reshape(-1)
            smooth_ = pd.concat((smooth_, smooth), axis=0)

    if write_csv:
        smooth_.to_csv(csv_filename[0], index=False)
        meta_.to_csv(csv_filename[1], index=False)
        light_.to_csv(csv_filename[2], index=False)

    # ade_ = sum(ade_bigls) / len(ade_bigls)
    # fde_ = sum(fde_bigls) / len(fde_bigls)

    motor_ade, motor_fde = (a1 / c1).item(), (f1 / c1).item()
    car_ade, car_fde = ((a2 + a3 + a5) / (c2 + c3 + c5)).item(), ((f2 + f3 + f5) / (c2 + c3 + f5)).item()
    biker_ade, biker_fde = (a4 / c4).item(), (f4 / c4).item()
    tricycle_ade, tricycle_fde = (a6 / c6).item(), (f6 / c6).item()
    print('Motor:\t\t', motor_ade, motor_fde)
    print('Car:\t\t', car_ade, car_fde)
    # print('Bus:\t\t', (a3 / c3).item(), (f3 / c3).item())
    print('Biker:\t\t', biker_ade, biker_fde)
    # print('Truck:\t\t', (a5 / c5).item(), (f5 / c5).item())
    print('Tricycle:\t', tricycle_ade, tricycle_fde)

    # print("ADE:", str(ade_).split('.')[0] + '.' + str(ade_).split('.')[1][0:2], "\tFDE:",
    #       str(fde_).split('.')[0] + '.' + str(fde_).split('.')[1][0:2], "\n")
    if no_pedestrian == False:
        pedestrian_ade, pedestrian_fde = (a0 / c0).item(), (f0 / c0).item()
        print('Pedestrian:\t', pedestrian_ade, pedestrian_fde)
        ade_ = (car_ade + biker_ade + tricycle_ade + motor_ade + pedestrian_ade) / 5
        fde_ = (car_fde + biker_fde + tricycle_fde + motor_fde + pedestrian_fde) / 5
        print("ADE: " + str(ade_) + "\tFDE: " + str(fde_))
        ade_ = sum(ade_bigls) / len(ade_bigls)
        fde_ = sum(fde_bigls) / len(fde_bigls)
        print("ADE: " + str(ade_) + "\tFDE: " + str(fde_))
    else:
        ade_ = (car_ade + biker_ade + tricycle_ade + motor_ade) / 4
        fde_ = (car_fde + biker_fde + tricycle_fde + motor_fde) / 4
        print("ADE: " + str(ade_) + "\tFDE: " + str(fde_))
        ade_ = sum(ade_bigls) / len(ade_bigls)
        fde_ = sum(fde_bigls) / len(fde_bigls)
        print("ADE: " + str(ade_) + "\tFDE: " + str(fde_))
    if use_scale:
        print('Test Time with Scaler (' + str(order) + '): ', str(datetime.now() - starttime))
    else:
        print('Test Time without Scaler (' + str(order) + '): ', str(datetime.now() - starttime))

    return ade_, fde_, raw_data_dict


from model import *
from pre_data_graph_sind import *

############# modified by necessary #############
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
obs_len, pred_len = 8, 12
paths = ['./checkpoint/' + str(obs_len) + '_' + str(pred_len) + '/sind']  # checkpoint path
low_bound = 1.0  # T parameter low bound
use_unified_bound = False  # use the low bound or use the trained parameter
num_sample = 5
it = 1
no_pedestrian = False  # metrics whitout considering pedestrians
csv_filename = ['smooth_' + str(obs_len) + '_' + str(pred_len) + '_multimodal.csv',
                'meta_' + str(obs_len) + '_' + str(pred_len) + '_multimodal.csv',
                'light_' + str(obs_len) + '_' + str(pred_len) + '_multimodal.csv']
write_csv = False  # the output is written as .csv
multimodal = True  # the best prediction path on .csv or all multimodal predcitions on .csv
############# modified by necessary #############

print("*" * 50)

for feta in range(len(paths)):

    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:', exps)

    for exp_path in exps:
        print("\n" + "*" * 50)
        print("Evaluating model:", exp_path)
        if use_unified_bound:
            lb = low_bound
        else:
            try:
                lb = float(exp_path.split('\\')[-2].split('_')[-1])
            except:
                lb = low_bound
        print("Low bound of T distribution: ", lb)
        model_path = exp_path + '/val_best.pth'
        args_path = exp_path + '/args.pkl'
        with open(args_path, 'rb') as stag_epoch:
            args = pickle.load(stag_epoch)
        print("paras:", args)

        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as stag_epoch:
            cm = pickle.load(stag_epoch)
        print("Stats:", cm)

        # Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len

        dset_test = dataset(
            './datasets_graph_' + str(obs_len) + '_' + str(
                pred_len) + '/' + args.dataset + '/test/' + args.dataset + '_test.npz')

        loader_test = DataLoader(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False)

        model = ddgc(n_spa=args.n_spa, n_cnn=args.n_cnn, n_gcn=args.n_gcn,
                     input=args.input_size, output=args.output_size, obs_len=args.obs_seq_len,
                     pred_len=args.pred_seq_len).cuda()

        model.load_state_dict(torch.load(model_path)['model'])

        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of Weights:', total_num)

        ade_noscale = []
        fde_noscale = []
        ade_scale = []
        fde_scale = []

        # print("Testing ....\n")

        for i in range(it):
            print()
            ad, fd, raw_data_dic_ = test(model, num_sample, lb, False, i + 1, write_csv, no_pedestrian, csv_filename,
                                         multimodal)
            ade_noscale.append(ad)
            fde_noscale.append(fd)
