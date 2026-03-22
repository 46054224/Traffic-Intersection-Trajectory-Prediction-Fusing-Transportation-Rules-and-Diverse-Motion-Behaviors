import numpy as np
import os
import torch
import pandas as pd
import copy


class dataset():
    def __init__(self, road, behavior, ped_traj, car_traj, light, train_rate, val_rate, frame_interval=1):
        self.train_rate, self.val_rate = train_rate, val_rate
        road = np.load(os.path.join(road))
        id = torch.from_numpy(road.get('ids')[:, np.newaxis])
        type = torch.from_numpy(road.get('types')[:, np.newaxis])
        self.id_type = torch.cat((id, type), -1)
        self.pos = torch.from_numpy(road.get('poss'))
        self.mask = torch.from_numpy(road.get('masks')).bool()
        self.frame_interval = frame_interval

        car_traj = pd.read_csv(car_traj)
        # keys: 'track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx',
        #        'vy', 'yaw_rad', 'heading_rad', 'length', 'width', 'ax', 'ay', 'v_lon',
        #        'v_lat', 'a_lon', 'a_lat'
        # save format: frame_id (ascend order), track_id, x, y, vx, vy, ax, ay, v_lon',
        #        'v_lat', 'a_lon', 'a_lat', 'yaw_rad', 'heading_rad', 'agent_type'
        fid = car_traj['frame_id'].to_numpy()[:, np.newaxis]
        id = car_traj['track_id'].to_numpy()[:, np.newaxis]
        t = car_traj['agent_type'].to_numpy()
        x = car_traj['x'].to_numpy()[:, np.newaxis]
        y = car_traj['y'].to_numpy()[:, np.newaxis]
        vx = car_traj['vx'].to_numpy()[:, np.newaxis]
        vy = car_traj['vy'].to_numpy()[:, np.newaxis]
        ax = car_traj['ax'].to_numpy()[:, np.newaxis]
        ay = car_traj['ay'].to_numpy()[:, np.newaxis]
        vlon = car_traj['v_lon'].to_numpy()[:, np.newaxis]
        vlat = car_traj['v_lat'].to_numpy()[:, np.newaxis]
        alon = car_traj['a_lon'].to_numpy()[:, np.newaxis]
        alat = car_traj['a_lat'].to_numpy()[:, np.newaxis]
        yaw = car_traj['yaw_rad'].to_numpy()[:, np.newaxis]
        head = car_traj['heading_rad'].to_numpy()[:, np.newaxis]
        # 'motorcycle':1, 'car':2, 'bus':3, 'bicycle':4, 'truck':5, 'tricycle':6
        for i in range(len(id)):
            if t[i] == 'motorcycle':
                t[i] = 1
            elif t[i] == 'car':
                t[i] = 2
            elif t[i] == 'bus':
                t[i] = 3
            elif t[i] == 'bicycle':
                t[i] = 4
            elif t[i] == 'truck':
                t[i] = 5
            elif t[i] == 'tricycle':
                t[i] = 6
        t = t.astype(np.float)[:, np.newaxis]
        car_traj = np.concatenate((fid, id, x, y, vx, vy, ax, ay, vlon, vlat, alon, alat, yaw, head, t), -1).astype(
            np.float)
        fid = car_traj[:, 0]
        index = np.argsort(fid).astype(np.int)
        car_traj = car_traj[index]
        max_id = int(np.max(car_traj[:, 1]))

        behavior = pd.read_csv(os.path.join(behavior))
        # 'trackId', 'initialFrame', 'finalFrame', 'Frame_nums', 'width',
        #        'length', 'class', 'CrossType', 'Signal_Violation_Behavior'
        # CrossType: StraightCross(0), LeftTurn(1), RightTurn(2), Others(3)
        # Signal_Violation_Behavior: No violation of traffic lights(0), yellow-light running(1), red-light running(2)

        self.nvi, self.yel, self.red, self.ped = 0, 0, 0, 0

        car_traj = np.concatenate((car_traj, np.ones((car_traj.shape[0], 2)) * -1), 1)
        for i in range(1, len(behavior) + 1):
            inform = behavior[behavior['trackId'] == i]
            agent = car_traj[car_traj[:, 1] == i]
            if (inform['CrossType'] == 'StraightCross').item():
                agent[:, -2] = 0
            elif (inform['CrossType'] == 'LeftTurns').item():
                agent[:, -2] = 1
            elif (inform['CrossType'] == 'RightTurn').item():
                agent[:, -2] = 2
            else:
                agent[:, -2] = 3
            if inform['Signal_Violation_Behavior'].values[0] == 'No violation of traffic lights':
                agent[:, -1] = 0
                self.nvi += 1
            elif inform['Signal_Violation_Behavior'].values[0] == 'yellow-light running ':
                agent[:, -1] = 1
                self.yel += 1
            else:
                agent[:, -1] = 2
                self.red += 1
            car_traj[car_traj[:, 1] == i] = agent
        # save format: 0) frame_id (ascend order), 1) track_id, 2) x, 3) y, 4) vx, 5) vy, 6) ax, 7) ay, 8) v_lon,
        #                 9) v_lat, 10) a_lon, 11) a_lat, 12) yaw_rad, 13) heading_rad, 14) agent_type, 15) CrossType,
        #                 16) Signal_Violation_Behavior

        ped_traj = pd.read_csv(ped_traj)
        # keys: 'track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx',
        #        'vy', 'ax', 'ay'
        fid = ped_traj['frame_id'][:, np.newaxis]
        id = ped_traj['track_id']
        t = ped_traj['agent_type']
        x = ped_traj['x'][:, np.newaxis]
        y = ped_traj['y'][:, np.newaxis]
        vx = ped_traj['vx'][:, np.newaxis]
        vy = ped_traj['vy'][:, np.newaxis]
        ax = ped_traj['ax'][:, np.newaxis]
        ay = ped_traj['ay'][:, np.newaxis]

        for i in range(len(id)):
            id[i] = int(id[i].split('P')[-1]) + max_id
            if t[i] == 'pedestrian':
                t[i] = 0
        self.ped = len(np.unique(id))
        id = id[:, np.newaxis]
        t = t[:, np.newaxis]
        ped_traj = np.concatenate((fid, id, x, y, vx, vy, ax, ay, t), -1).astype(np.float)
        fid = ped_traj[:, 0]
        index = np.argsort(fid).astype(np.int)
        ped_traj = ped_traj[index]

        pt = np.zeros((ped_traj.shape[0], car_traj.shape[1]))
        pt[:, :ped_traj.shape[1]] = ped_traj
        pt[:, -2:] = -1
        pt = np.concatenate((car_traj, pt), 0)
        fid = pt[:, 0]
        index = np.argsort(fid).astype(np.int)

        self.pt = torch.from_numpy(pt[index])
        max_frame = int(torch.max(self.pt[:, 0]).item())
        self.traj_list = []
        frame_list = self.pt[:, 0]

        light = pd.read_csv(os.path.join(light))
        # 'RawFrameID' / 3 = Frame, 'timestamp(ms)', 'Traffic light 1', 'Traffic light 2',
        #        'Traffic light 3', 'Traffic light 4', 'Traffic light 5',
        #        'Traffic light 6', 'Traffic light 7', 'Traffic light 8'
        frame_id = (light['RawFrameID'].to_numpy() / 3).astype(np.float)
        frame_id[0] = 0
        frame_id = frame_id.repeat(2)
        frame_id = frame_id.reshape((int(len(frame_id) / 2), 2))
        frame_id[:-1, 1] = frame_id[1:, 1]
        frame_id[-1, 1] = max_frame + 1
        tls = light['Traffic light 1'].to_numpy().astype(np.int)
        tls = tls.repeat(10)
        tls = tls.reshape((int(len(tls) / 10), 10))
        for i in range(1, 7):
            tls[:, i] = light['Traffic light ' + str(i + 1)].to_numpy().astype(np.int)
        tls[:, -2:] = frame_id
        j = 0

        for i in range(0, max_frame + 1, self.frame_interval):
            frame = self.pt[frame_list == i].numpy()
            if tls[j, -2] <= i < tls[j, -1]:
                ls = np.tile(tls[j, :8], frame.shape[0]).reshape(frame.shape[0], 8)
                frame = np.concatenate((frame[:, :14], ls, frame[:, 14:]), axis=-1)
            else:
                j += 1
                ls = np.tile(tls[j, :8], frame.shape[0]).reshape(frame.shape[0], 8)
                frame = np.concatenate((frame[:, :14], ls, frame[:, 14:]), axis=-1)
            self.traj_list.append(frame)
        # save format: 0) frame_id (ascend order), 1) track_id, 2) x, 3) y, 4) vx, 5) vy, 6) ax, 7) ay, 8) v_lon,
        #                 9) v_lat, 10) a_lon, 11) a_lat, 12) yaw_rad, 13) heading_rad, 14) light 0, 15) light 1,
        #                 16) light 2, 17) light 3, 18) light 4, 19) light 5, 20) light 6, 21) light 7, 22) agent_type,
        #                 23) CrossType, 24) Signal_Violation_Behavior

    def __getitem__(self, idx):
        out = [
            self.id_type,
            self.pos,
            self.mask,
            self.pt[idx],
            self.traj_list[idx]
        ]
        return out

    def __len__(self):
        return self.pt.shape[0]

    def getTrajs(self):
        return self.traj_list

    def wirte_by_frame(self, obs, pred):
        self.obs_len = obs
        self.pred_len = pred

        print('Frame Sample Interval: ', self.frame_interval)
        print('No violation of traffic lights in Dataset: ', self.nvi)
        print('Yellow-light running in Dataset: ', self.yel)
        print('Red-light running in Dataset: ', self.red)
        print('Pedestrian count in Dataset: ', self.ped)

        frame_data = self.getTrajs()
        max_frame = int(frame_data[-1][-1, 0] / self.frame_interval) + 1
        max_frame_seg = int(max_frame / (self.obs_len + self.pred_len))
        print('Num of Frames in Dataset: ' + str(max_frame) + '\nSeg of Frames in Dataset: ' + str(max_frame_seg))

        train_frame_bound = int(max_frame * self.train_rate)
        val_frame_bound = int(max_frame * self.val_rate)
        train_data = frame_data[:train_frame_bound]
        train_frame = len(train_data)
        max_frame_seg = int(train_frame / (self.obs_len + self.pred_len))
        print()

        print('Mode: Train\nNum of Frames: ' + str(
            train_frame) + '\nSeg of Frames: ' + str(max_frame_seg))
        file = open('sind_train.txt', 'w+')
        nvi, yel, red, ped = 0, 0, 0, 0
        save_ids = np.zeros((0))
        for traj in train_data:
            ids = np.unique(traj[:, 1])
            for id in ids:
                if id not in save_ids:
                    t = traj[traj[:, 1] == id][0, -1]
                    if t == 0:
                        nvi += 1
                    elif t == 1:
                        yel += 1
                    elif t == 2:
                        red += 1
                    else:
                        ped += 1
                    save_ids = np.concatenate((save_ids, np.array([id])), axis=0)
            traj = traj.astype(np.str)
            for t in traj:
                for e in t:
                    file.write(e)
                    file.write('\t')
                file.write('\n')
        file.close()
        print('No violation of traffic lights in Train: ', nvi)
        print('Yellow-light running in Train: ', yel)
        print('Red-light running in Train: ', red)
        print('Pedestrian count in Train: ', ped)
        print()

        val_data = frame_data[train_frame_bound:train_frame_bound + val_frame_bound]
        val_frame = len(val_data)
        max_frame_seg = int(val_frame / (self.obs_len + self.pred_len))
        print('Mode: Valuation\nNum of Frames: ' + str(
            val_frame) + '\nSeg of Frames: ' + str(max_frame_seg))
        file = open('sind_val.txt', 'w+')
        nvi, yel, red, ped = 0, 0, 0, 0
        save_ids = np.zeros((0))
        for traj in val_data:
            ids = np.unique(traj[:, 1])
            for id in ids:
                if id not in save_ids:
                    t = traj[traj[:, 1] == id][0, -1]
                    if t == 0:
                        nvi += 1
                    elif t == 1:
                        yel += 1
                    elif t == 2:
                        red += 1
                    else:
                        ped += 1
                    save_ids = np.concatenate((save_ids, np.array([id])), axis=0)
            traj = traj.astype(np.str)
            for t in traj:
                for e in t:
                    file.write(e)
                    file.write('\t')
                file.write('\n')
        file.close()
        print('No violation of traffic lights in Val: ', nvi)
        print('Yellow-light running in Val: ', yel)
        print('Red-light running in Val: ', red)
        print('Pedestrian count in Val: ', ped)
        print()

        test_data = frame_data[train_frame_bound + val_frame_bound:]
        test_frame = len(test_data)
        max_frame_seg = int(test_frame / (self.obs_len + self.pred_len))
        print('Mode: Valuation\nNum of Frames: ' + str(
            test_frame) + '\nSeg of Frames: ' + str(max_frame_seg))
        file = open('sind_test.txt', 'w+')
        nvi, yel, red, ped = 0, 0, 0, 0
        save_ids = np.zeros((0))
        for traj in test_data:
            ids = np.unique(traj[:, 1])
            for id in ids:
                if id not in save_ids:
                    t = traj[traj[:, 1] == id][0, -1]
                    if t == 0:
                        nvi += 1
                    elif t == 1:
                        yel += 1
                    elif t == 2:
                        red += 1
                    else:
                        ped += 1
                    save_ids = np.concatenate((save_ids, np.array([id])), axis=0)
            traj = traj.astype(np.str)
            for t in traj:
                for e in t:
                    file.write(e)
                    file.write('\t')
                file.write('\n')
        file.close()
        print('No violation of traffic lights in Test: ', nvi)
        print('Yellow-light running in Test: ', yel)
        print('Red-light running in Test: ', red)
        print('Pedestrian count in Test: ', ped)
        print()


if __name__ == "__main__":
    frame_interval = 1
    data = dataset('road_information.npz', 'Veh_tracks_meta.csv', 'Ped_smoothed_tracks.csv',
                   'Veh_smoothed_tracks.csv', 'TrafficLight_8_02_1.csv', train_rate=0.6, val_rate=0.2,
                   frame_interval=frame_interval)
    data.wirte_by_frame(8, 12)
    print()
