import os
import math
import scipy.sparse as sp
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sind_dataset import dataset as SIND
import copy


# action space [stop, straight, left, right]

# lane bounds
# motions in these ranges need observe traffic lights
# when the light is red/yellow(green), recommend actions should stop/right (straight/left/right).
# light index 0, 1, 2, 3, 4, 5, 6, 7
# light 2: y<16&x<4.5
# light 4: x>14.6&y<5.8
# light 0: x<12.8&y>26.1
# light 6: x>24.9&y>16.4
# in these ranges straight should be the unique candidate action
# 1)x>=4.5&x<14.6&y<5.8
# 2)y>=16&y<26.1&x<4.5
# 3)y>=5.8&y<16.4&x>24.9
# 4)x>=12.8&x<24.9&y>26.1
# otehrwise: this range is the crossroad center, the candidate action cannot be stop.

def find_fields(obs):
    # index 0 is in the crossroad, 1, 3, 5, 7 are the stop fields, 1+1, 3+1, 5+1, 7+1 are the running fields.
    #           |   | ↑ |
    #     light0|   |   |light7
    #     light1|   |   |light6
    #   ________| 1 | 2 |________
    # ←8________    0    ________3
    #  7________         ________4→
    #     light2|   |   |light5
    #     light3|   |   |light4
    #           | 6 | 5 |
    #           | ↓ |   |

    # obs/pred.shape [num_agents, 23, num_steps]
    # {0: 'red', 1: 'green', 3: 'yellow'}
    # [stop, straight, left, right]
    num_agents = obs.shape[0]
    lights = obs[:, 12:20, :]
    light = torch.ones((num_agents)) * -1  # index 0
    ########## agent in someone fields ##########
    fields = torch.zeros((num_agents, obs.shape[-1]))
    main_field = torch.zeros((num_agents))
    # index 0 is in the crossroad, 1, 3, 5, 7 are the stop fields, 1+1, 3+1, 5+1, 7+1 are the running fields.
    x, y = obs[:, 0], obs[:, 1]
    for i in range(num_agents):
        for j in range(obs.shape[-1]):
            if x[i, j] < 4.5 and y[i, j] < 16:
                fields[i, j] = 7
            if x[i, j] > 14.6 and y[i, j] < 5.8:
                fields[i, j] = 5
            if x[i, j] < 12.8 and y[i, j] > 26.1:
                fields[i, j] = 1
            if x[i, j] > 24.9 and y[i, j] > 16.4:
                fields[i, j] = 3
            if x[i, j] >= 4.5 and x[i, j] < 14.6 and y[i, j] < 5.8:
                fields[i, j] = 6
            if x[i, j] < 4.5 and y[i, j] >= 16 and y[i, j] < 26.1:
                fields[i, j] = 8
            if x[i, j] > 24.9 and y[i, j] >= 5.8 and y[i, j] < 16.4:
                fields[i, j] = 4
            if x[i, j] >= 12.8 and x[i, j] < 24.9 and y[i, j] > 26.1:
                fields[i, j] = 2
        if fields[i, 0] == 0 and fields[i, -1] == 0:
            main_field[i] = 0
        if fields[i, -1] != 0:
            main_field[i] = fields[i, -1]
            if fields[i, -1] == 7:
                light[i] = lights[i, 2, -1]
            if fields[i, -1] == 5:
                light[i] = lights[i, 4, -1]
            if fields[i, -1] == 1:
                light[i] = lights[i, 0, -1]
            if fields[i, -1] == 3:
                light[i] = lights[i, 6, -1]
        if fields[i, -1] == 0 and fields[i, 0] != 0:
            main_field[i] = fields[i, 0]
            if fields[i, 0] == 7:
                light[i] = lights[i, 2, 0]
            if fields[i, 0] == 5:
                light[i] = lights[i, 4, 0]
            if fields[i, 0] == 1:
                light[i] = lights[i, 0, 0]
            if fields[i, 0] == 3:
                light[i] = lights[i, 6, 0]
    ########## agent in someone range ##########
    return fields, main_field, light


def true_action(obs, pred):
    ########## true actions for whole trajectories ##########
    # obs/pred.shape [num_agents, 23, num_steps]
    # {0: 'red', 1: 'green', 3: 'yellow'}
    # [stop, straight, left, right]
    num_agents = obs.shape[0]
    action_type = torch.zeros((num_agents, 2))  # [left and straight, right and stop]
    action_type1 = torch.zeros((num_agents, 4))  # [stop, right, straight, left]
    trajs = torch.cat((obs, pred), dim=-1)[:, 0:2]  # [num_agents, 2, obs_len+pred_len]
    a = trajs[:, :, -1] - trajs[:, :, 0]
    b = trajs[:, :, obs.shape[-1] - 1] - trajs[:, :, 0]
    angle = torch.acos(torch.sum(a.mul(b), dim=-1) / (
        torch.sqrt(torch.sum((a ** 2), dim=-1)).mul(torch.sqrt(torch.sum((b ** 2), dim=-1)))))
    distance = torch.sqrt(torch.sum((a - b) ** 2, dim=-1))
    # A = np.array([1, 0, 0, 1])  # stop/right
    # B = np.array([0, 1, 1, 1])  # straight/left/right
    # C = np.array([0, 1, 0, 0])  # straight
    for i in range(num_agents):
        if distance[i] > 0.5 or torch.isnan(angle[i]) == False:
            if angle[i] > np.pi / 6:  # left or right
                oritation_point = trajs[i, :, 0]
                goal_point = trajs[i, :, -1]
                right_point = copy.deepcopy(goal_point)
                left_point = copy.deepcopy(goal_point)
                left_point[0] = (goal_point[0] - oritation_point[0]) * torch.cos(angle[i]) + (
                        goal_point[1] - oritation_point[1]) * (
                                    -torch.sin(angle[i])) + oritation_point[0]
                left_point[1] = (goal_point[0] - oritation_point[0]) * torch.sin(angle[i]) + (
                        goal_point[1] - oritation_point[1]) * (
                                    torch.cos(angle[i])) + oritation_point[1]

                right_point[0] = (goal_point[0] - oritation_point[0]) * torch.cos(-angle[i]) + (
                        goal_point[1] - oritation_point[1]) * (
                                     -torch.sin(-angle[i])) + oritation_point[0]
                right_point[1] = (goal_point[0] - oritation_point[0]) * torch.sin(-angle[i]) + (
                        goal_point[1] - oritation_point[1]) * (
                                     torch.cos(-angle[i])) + oritation_point[1]
                obs_point = trajs[i, :, obs.shape[-1] - 1]
                if torch.sum((obs_point - right_point) ** 2, dim=-1) < torch.sum((obs_point - left_point) ** 2, dim=-1):
                    action_type[i, 1] = 1  # right
                    action_type1[i, 1] = 1  # right
                else:
                    action_type[i, 0] = 1  # left
                    action_type1[i, 3] = 1  # left
            else:
                action_type[i, 0] = 1  # straight
                action_type1[i, 2] = 1  # straight
        else:
            action_type[i, 1] = 1  # stop
            action_type1[i, 0] = 1  # stop
    ########## true actions for whole trajectories ##########
    return action_type, action_type1


def recommend_actions(obs, main_field, light):
    ########## candicated actions based on history traffic lights ##########
    # index 0 is in the crossroad, 1, 3, 5, 7 are the stop fields, 1+1, 3+1, 5+1, 7+1 are the running fields.
    #           |   | ↑ |
    #     light0|   |   |light7
    #     light1|   |   |light6
    #   ________| 1 | 2 |________
    # ←8________    0    ________3
    #  7________         ________4→
    #     light2|   |   |light5
    #     light3|   |   |light4
    #           | 6 | 5 |
    #           | ↓ |   |

    num_agents = obs.shape[0]
    obs = obs[:, :2]  # [num_agnets, 2, obs_len]
    actions = torch.zeros((num_agents, 2, obs.shape[-1], 5))  # 0original, 1stop, 2straight, 3left, 4right
    actions[:, :, :, 0] = obs  # 0original
    motion_direction = torch.mean((obs[:, :, 1:] - obs[:, :, :-1]), dim=-1) + 1e-10
    lane_direction = torch.zeros((4, 2))  # ↑↓←→
    lane_direction[0, 1], lane_direction[1, 1], lane_direction[2, 0], lane_direction[3, 0] = 1, -1, -1, 1
    for i in range(num_agents):
        if main_field[i] == 1 or main_field[i] == 6:
            selected_direction = lane_direction[1]
        if main_field[i] == 2 or main_field[i] == 5:
            selected_direction = lane_direction[0]
        if main_field[i] == 3 or main_field[i] == 8:
            selected_direction = lane_direction[2]
        if main_field[i] == 4 or main_field[i] == 7:
            selected_direction = lane_direction[3]

        if main_field[i] != 0:
            angle = torch.acos(torch.sum(motion_direction[i].mul(selected_direction)) /
                               (torch.sqrt(torch.sum(motion_direction[i] ** 2)) * torch.sqrt(
                                   torch.sum(selected_direction ** 2))))
        else:
            md = motion_direction[i].unsqueeze(0).repeat(4, 1)
            angles = torch.acos(torch.sum(md.mul(lane_direction)) /
                                (torch.sqrt(torch.sum(md ** 2)) * torch.sqrt(
                                    torch.sum(selected_direction ** 2))))
            angle = angles[torch.argmin(angles)]
        o = obs[i, :, 0:1].repeat(1, obs.shape[-1])  # [2, obs_len]
        ax = (obs[i, 0, :] - o[0, :]) * torch.cos(angle) + (
                obs[i, 1, :] - o[1, :]) * (-torch.sin(angle)) + o[0, :]
        ay = (obs[i, 0, :] - o[0, :]) * torch.sin(angle) + (
                obs[i, 1, :] - o[1, :]) * (torch.cos(angle)) + o[1, :]
        bx = (obs[i, 0, :] - o[0, :]) * torch.cos(angle) + (
                obs[i, 1, :] - o[1, :]) * (-torch.sin(angle)) + o[0, :]
        by = (obs[i, 0, :] - o[0, :]) * torch.sin(angle) + (
                obs[i, 1, :] - o[1, :]) * (torch.cos(angle)) + o[1, :]
        adis = torch.mean(torch.sqrt(torch.sum((ax - obs) ** 2, 0)))
        bdis = torch.mean(torch.sqrt(torch.sum((bx - obs) ** 2, 0)))
        if adis > bdis:
            lane_obs = torch.cat((bx.unsqueeze(0), by.unsqueeze(0)), dim=0)  # [2, obs_len]
        else:
            lane_obs = torch.cat((ax.unsqueeze(0), ay.unsqueeze(0)), dim=0)  # [2, obs_len]
        # the fields can consider less (ignore) about lights
        if main_field[i] == 2 or main_field[i] == 4 or main_field[i] == 6 or main_field[i] == 8 or main_field[i] == 0:
            actions[i, :, :, 2] = lane_obs  # 2straight
        # {0: 'red', 1: 'green', 3: 'yellow'}
        # light[i] == 1, green light, there are all the acitions, stop, straight, left, and right
        if (main_field[i] == 1 or main_field[i] == 3 or main_field[i] == 5 or main_field[i] == 7) and light[i] == 1:
            actions[i, :, :, 1] = obs[i, :, -1:].repeat(1, obs.shape[-1])  # 1stop
            actions[i, :, :, 2] = lane_obs  # 2straight
            # 3left
            actions[i, 0, :, 3] = (lane_obs[0, :] - o[0, :]) * torch.cos(torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (-torch.sin(torch.Tensor([np.pi / 6]) / 6)) + o[0, :]
            actions[i, 1, :, 3] = (lane_obs[0, :] - o[0, :]) * torch.sin(torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (torch.cos(torch.Tensor([np.pi / 6]) / 6)) + o[1, :]
            # 4right
            actions[i, 0, :, 4] = (lane_obs[0, :] - o[0, :]) * torch.cos(-torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (-torch.sin(-torch.Tensor([np.pi / 6]) / 6)) + o[0, :]
            actions[i, 1, :, 4] = (lane_obs[0, :] - o[0, :]) * torch.sin(-torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (torch.cos(-torch.Tensor([np.pi / 6]) / 6)) + o[1, :]
        # the light is red or yellow
        if (main_field[i] == 1 or main_field[i] == 3 or main_field[i] == 5 or main_field[i] == 7) and light[i] != 1:
            actions[i, :, :, 1] = obs[i, :, -1:].repeat(1, obs.shape[-1])  # 1stop
            # 4right
            actions[i, 0, :, 4] = (lane_obs[0, :] - o[0, :]) * torch.cos(-torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (-torch.sin(-torch.Tensor([np.pi / 6]) / 6)) + o[0, :]
            actions[i, 1, :, 4] = (lane_obs[0, :] - o[0, :]) * torch.sin(-torch.Tensor([np.pi / 6]) / 6) + (
                    lane_obs[1, :] - o[1, :]) * (torch.cos(-torch.Tensor([np.pi / 6]) / 6)) + o[1, :]
    ########## candicated actions based on history traffic lights ##########
    return actions


def generate_actions(obs, pred):
    # obs/pred.shape [num_agents, 23, num_steps]
    # {0: 'red', 1: 'green', 3: 'yellow'}
    # [stop, straight, left, right]
    # num_agents = obs.shape[0]

    fields, main_field, light = find_fields(obs)
    action_traj = recommend_actions(obs, main_field, light)
    action_type, action_type1, = true_action(obs, pred)

    return action_traj.numpy(), action_type.numpy(), action_type1.numpy(), fields.numpy()


def view(p1, p2):
    NORM = (math.sqrt((p1[0] ** 2 + p1[1] ** 2)) * math.sqrt((p2[0] ** 2 + p2[1] ** 2)))
    if NORM == 0:
        return 1
    else:
        NORM = (p1[0] * p2[0] + p1[1] * p2[1]) / NORM

        if NORM > 1:
            NORM = 1

        if NORM < 0:
            NORM = 0
        else:
            NORM = 1
        return NORM


def direction(d1, d2, d12):
    # d1 is the directional vecctor of objective agent

    if torch.sum(abs(d1[0]) + abs(d1[1])) == 0 or torch.sum(abs(d2[0]) + abs(d2[1])) == 0. or \
            torch.sum(abs(d12[0]) + abs(d12[1])) == 0.:
        return 1
    else:
        d12 = -d12

        a_d1_d12 = (d1[0] * d12[0] + d1[1] * d12[1]) / \
                   (math.sqrt((d1[0] ** 2 + d1[1] ** 2)) * math.sqrt((d12[0] ** 2 + d12[1] ** 2)))

        if a_d1_d12 > 1:
            a_d1_d12 = 1
        elif a_d1_d12 < -1:
            a_d1_d12 = -1

        a_d2_d12 = (d2[0] * d12[0] + d2[1] * d12[1]) / \
                   (math.sqrt((d2[0] ** 2 + d2[1] ** 2)) * math.sqrt((d12[0] ** 2 + d12[1] ** 2)))

        if a_d2_d12 > 1:
            a_d2_d12 = 1
        elif a_d2_d12 < -1:
            a_d2_d12 = -1

        a_d1_d2 = (d1[0] * d2[0] + d1[1] * d2[1]) / \
                  (math.sqrt((d1[0] ** 2 + d1[1] ** 2)) * math.sqrt((d2[0] ** 2 + d2[1] ** 2)))

        if a_d1_d2 > 1:
            a_d1_d2 = 1
        elif a_d1_d2 < -1:
            a_d1_d2 = -1

        a_d1_d2 = math.acos(a_d1_d2)
        a_d1_d12 = math.acos(a_d1_d12)
        a_d2_d12 = math.acos(a_d2_d12)

        if abs(a_d1_d12 - (a_d2_d12 + a_d1_d2)) < 1e-6:
            return 1
        else:
            return 0


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) + 1
    return 1 / NORM


def anorm_speed(speed):
    NORM = math.sqrt(speed[0] ** 2 + speed[1] ** 2)
    return math.tanh(NORM)


def seq_to_graph(seq_, seq_rel, fields=None):
    # agents in fileds [num_agent, num_steps], no fields for future features (no graph).
    type_ = False
    if seq_.shape[1] != 2:
        type_ = True
        type = seq_[:, 20, 0]
    # save format: 0) frame_id (ascend order) <mask>, 1) track_id <mask>, 2) x, 3) y, 4) vx, 5) vy, 6) ax, 7) ay,
    #              8) v_lon, 9) v_lat, 10) a_lon, 11) a_lat, 12) yaw_rad, 13) heading_rad, 14) light 0, 15) light 1,
    #                 16) light 2, 17) light 3, 18) light 4, 19) light 5, 20) light 6, 21) light 7, 22) agent_type,
    #                 23) CrossType, 24) Signal_Violation_Behavior

    # seq_, seq_rel (num_agents, 23, num_steps)

    if fields is not None:
        seq_, seq_rel = seq_[:, :-2], seq_rel[:, :-2]
    else:
        # seq_, seq_rel = torch.cat((seq_[:, :2], seq_[:, -3:]), dim=1), torch.cat(
        #     (seq_rel[:, :2], seq_rel[:, -3:]), dim=1)
        seq_, seq_rel = torch.cat((seq_[:, :2], seq_[:, -11:]), dim=1), torch.cat(
            (seq_rel[:, :2], seq_rel[:, -11:]), dim=1)

    max_nodes, xy, seq_len = seq_.shape[0], seq_.shape[1], seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, xy))
    A = np.zeros((seq_len, max_nodes, max_nodes, 4))

    for s in range(seq_len):
        step_rel = seq_rel[:, :, s]
        for h in range(max_nodes):
            V[s, h, :] = step_rel[h]
            if fields is not None:
                for i in range(max_nodes):
                    if i == h:
                        A[s, h, h] = 1
                    else:
                        # agents in fileds [num_agent, num_steps], no fields for future features (no graph).
                        # index 0 is in the crossroad, 1, 3, 5, 7 are the stop fields, 1+1, 3+1, 5+1, 7+1 are the running fields.
                        # 0-1-3-5-7, 1-2-0, 2-1
                        if type[i] == 0:  # the h-th agent is influenced by the i-th agents
                            A[s, i, h, 3] = anorm_speed(step_rel[i]) / (
                                    math.tanh(anorm(seq_[i, :, s], seq_[h, :, s])) + 1)  # field
                        elif fields[h, s] == 0 and (
                                fields[i, s] == 1 or fields[i, s] == 3 or fields[i, s] == 5 or fields[i, s] == 7 or
                                fields[i, s] == 0):
                            A[s, i, h, 3] = anorm_speed(step_rel[i]) / (
                                    math.tanh(anorm(seq_[i, :, s], seq_[h, :, s])) + 1)  # field
                        elif fields[h, s] % 2 == 1 and (
                                fields[i, s] == fields[h, s] or fields[i, s] == fields[h, s] + 1 or fields[i, s] == 0):
                            A[s, i, h, 3] = anorm_speed(step_rel[i]) / (
                                    math.tanh(anorm(seq_[i, :, s], seq_[h, :, s])) + 1)  # field
                        elif fields[h, s] % 2 == 0 and fields[h, s] != 0 and (
                                fields[i, s] == fields[h, s] or fields[i, s] == fields[h, s] - 1):
                            A[s, i, h, 3] = anorm_speed(step_rel[i]) / (
                                    math.tanh(anorm(seq_[i, :, s], seq_[h, :, s])) + 1)  # field

                        if s == seq_len - 1:
                            d1 = seq_[h, :2, s] - seq_[h, :2, s - 1]  # current agent
                            d12 = seq_[i, :2, s] - seq_[h, :2, s]
                            d2 = seq_[i, :2, s] - seq_[i, :2, s - 1]

                            a = anorm(seq_[i, :, s], seq_[h, :, s])

                            # SDD:Pedestrian1;Biker2;Skater3;Car4;Cart5;Bus6;
                            # SIND:Pedestrian0;Motor1;Car2;Bus3;Biker4;Track5;tricycle;
                            if (type_ == False) or (type_ == True and (type[h] in [0, 4])):
                                b = view(d1, d12)
                            else:
                                b = 1
                            A[s, i, h, 0] = a * b  # view, the h-th agent is influenced by the i-th agents

                            if b == 0:
                                A[s, i, h, 1] = 0  # direction
                            else:
                                c = direction(d1, d2, d12)
                                A[s, i, h, 1] = a * c  # direction
                                if c == 0:
                                    A[s, i, h, 2] = 0  # rate
                                else:
                                    A[s, i, h, 2] = anorm_speed(step_rel[i])  # rate
                        else:
                            d1 = seq_[h, :2, s + 1] - seq_[h, :2, s]
                            d12 = seq_[i, :2, s] - seq_[h, :2, s]
                            d2 = seq_[i, :2, s + 1] - seq_[i, :2, s]

                            a = anorm(seq_[i, :, s], seq_[h, :, s])

                            # Pedestrian1;Biker2;Skater3;Car4;Cart5;Bus6;
                            if (type_ == False) or (type_ == True and (type[h] in [0, 4])):
                                b = view(d1, d12)
                            else:
                                b = 1
                            A[s, i, h, 0] = a * b

                            if b == 0:
                                A[s, i, h, 1] = 0
                            else:
                                c = direction(d1, d2, d12)
                                A[s, i, h, 1] = a * c
                                if c == 0:
                                    A[s, i, h, 2] = 0
                                else:
                                    A[s, i, h, 2] = anorm_speed(step_rel[i])

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))

    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.

    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.__mul__(adj_mx)

    return random_walk_mx


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, data_dir_output, filename, obs_len=8, pred_len=8, skip=1, min_ped=1, delim='\t',
            norm_lap_matr=True, group_distance=1, frame_interval=1):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.filename = filename
        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.data_dir_output = data_dir_output
        self.gd = group_distance

        self.sind_dataset = SIND('road_information.npz', 'Veh_tracks_meta.csv', 'Ped_smoothed_tracks.csv',
                                 'Veh_smoothed_tracks.csv', 'TrafficLight_8_02_1.csv', train_rate=0.6, val_rate=0.2,
                                 frame_interval=frame_interval)

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []

        frame_data = self.sind_dataset.getTrajs()
        max_frame = int(frame_data[-1][-1, 0] / frame_interval) + 1
        max_frame_seg = int(max_frame / (self.obs_len + self.pred_len))
        print('Frame Sample Interval: ', frame_interval)
        print('Num of Frames in Dataset: ' + str(max_frame) + '\nSeg of Frames in Dataset: ' + str(max_frame_seg))
        print()

        train_frame_bound = int(max_frame * self.sind_dataset.train_rate)
        val_frame_bound = int(max_frame * self.sind_dataset.val_rate)
        # test_frame_bound = int(max_frame * 0.15)
        if filename.split('_')[-1].split('.')[0] == 'train':
            lowbound = 0
            upbound = train_frame_bound
        elif filename.split('_')[-1].split('.')[0] == 'val':
            lowbound = train_frame_bound
            upbound = train_frame_bound + val_frame_bound
        elif filename.split('_')[-1].split('.')[0] == 'test':
            lowbound = train_frame_bound + val_frame_bound
            upbound = max_frame
        else:
            print('incorrect mode')
            return
        frame_data = frame_data[lowbound:upbound]
        max_frame = len(frame_data)
        max_frame_seg = int(max_frame / (self.obs_len + self.pred_len))
        print('Mode: ' + filename.split('_')[-1].split('.')[0] + '\nNum of Frames: ' + str(
            max_frame) + '\nSeg of Frames: ' + str(max_frame_seg))
        frames = []
        for i in range(max_frame):
            frames.append(frame_data[i][0, 0])

        for idx in range(len(frame_data)):
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0)
            curr_seq_all = curr_seq_data
            curr_seq_data = curr_seq_data[:, :4]
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

            curr_seq_rel = np.zeros((len(peds_in_curr_seq), curr_seq_all.shape[-1] - 2, self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), curr_seq_all.shape[-1] - 2, self.seq_len))

            num_peds_considered = 0
            for _, ped_id in enumerate(peds_in_curr_seq):

                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                curr_ped_all = curr_seq_all[curr_seq_all[:, 1] == ped_id, :]
                if curr_ped_seq.shape[0] == self.seq_len:
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    curr_ped_all = np.around(curr_ped_all, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_all = np.transpose(curr_ped_all[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_all
                    curr_seq_rel[_idx, :, pad_front:pad_end] = copy.deepcopy(curr_ped_all)
                    curr_seq_rel[_idx, :2, pad_front:pad_end] = rel_curr_ped_seq
                    num_peds_considered += 1

            if num_peds_considered > min_ped:
                num_peds_in_seq.append(num_peds_considered)
                seq = curr_seq[:num_peds_considered]
                seq_rel = curr_seq_rel[:num_peds_considered]

                # save format: 0) frame_id (ascend order) <mask>, 1) track_id <mask>, 2) x, 3) y, 4) vx, 5) vy, 6) ax, 7) ay,
                #              8) v_lon, 9) v_lat, 10) a_lon, 11) a_lat, 12) yaw_rad, 13) heading_rad, 14) light 0, 15) light 1,
                #                 16) light 2, 17) light 3, 18) light 4, 19) light 5, 20) light 6, 21) light 7, 22) agent_type,
                #                 23) CrossType, 24) Signal_Violation_Behavior

                seq_list.append(seq)  # shape: agents, feats, seqlen
                seq_list_rel.append(seq_rel)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.gt_action = []
        self.gt_action1 = []
        self.ractions = []
        # self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        maxd = 0

        num_agent = []

        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            recommend_actions, true_action_type, true_action_type1, fields = generate_actions(
                obs=self.obs_traj_rel[start:end],
                pred=self.pred_traj_rel[start:end])
            # recommend_actions [num_agents, xy, num_steps, actions]
            # true_action_type [num_agents] 0: stop, 1: straight, 2: left, 3: right
            self.gt_action.append(true_action_type)
            self.gt_action1.append(true_action_type1)
            self.ractions.append(recommend_actions)

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], fields)
            v_ = v_.numpy()
            a_ = a_.numpy()
            self.v_obs.append(v_)
            self.A_obs.append(a_)

            v_, _ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
            v_ = v_.numpy()
            self.v_pred.append(v_)

            if v_.shape[1] > maxd:
                maxd = v_.shape[1]
            num_agent.append(v_.shape[1])

        v_obs = np.zeros((self.v_obs.__len__(), self.v_obs[0].shape[0], maxd, self.v_obs[0].shape[2]))
        # [scenes, steps, agents, feats]
        A_obs = np.zeros((self.A_obs.__len__(), self.A_obs[0].shape[0], maxd, maxd, 4))
        # [scenes, steps, agents, agents, vdr]
        v_pred = np.zeros((self.v_pred.__len__(), self.v_pred[0].shape[0], maxd, self.v_pred[0].shape[2]))
        # [scenes, steps, agents, feats]
        ractions = np.zeros((self.ractions.__len__(), self.ractions[0].shape[2], maxd, self.ractions[0].shape[1],
                             self.ractions[0].shape[3]))
        # [scenes, steps, agents, xy, actions]
        gt_action = np.zeros((self.gt_action.__len__(), maxd, 2))
        gt_action1 = np.zeros((self.gt_action1.__len__(), maxd, 4))
        # [scenes, agents]
        for i in range(self.v_obs.__len__()):
            A_obs[i, :, :self.A_obs[i].shape[1], :self.A_obs[i].shape[1], :] = self.A_obs[i]
            v_obs[i, :, :self.v_obs[i].shape[1], :] = self.v_obs[i]
            v_pred[i, :, :self.v_pred[i].shape[1], :] = self.v_pred[i]
            gt_action[i, :self.gt_action[i].shape[0]] = self.gt_action[i]
            gt_action1[i, :self.gt_action1[i].shape[0]] = self.gt_action1[i]
            for j in range(self.ractions[i].shape[2]):
                ractions[i, j, :self.ractions[i].shape[0]] = self.ractions[i][:, :, j]
            # [scenes, steps, agents, xy, actions], [num_agents, xy, num_steps, actions]

        np.savez_compressed(
            os.path.join(self.data_dir_output + self.filename),
            obs_traj=self.obs_traj,
            v_obs=v_obs,
            A_obs=A_obs,
            v_pred=v_pred,
            gt_action=gt_action,
            gt_action1=gt_action1,
            ractions=ractions,
            num_agent=np.array(num_agent),
            se=self.seq_start_end
        )

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index]

        ]
        return out


class dataset():
    def __init__(self, data):
        data = np.load(os.path.join(data))
        self.obs_traj = torch.from_numpy(data['obs_traj'])

        self.num_agent = torch.from_numpy(data['num_agent'])
        self.se = torch.from_numpy(data['se'])

        v_obs = torch.from_numpy(data['v_obs'])
        A_obs = torch.from_numpy(data['A_obs'])
        v_pred = torch.from_numpy(data['v_pred'])
        # A_pred = torch.from_numpy(data['A_pred'])

        ractions = torch.from_numpy(data['ractions'])
        gt_action = torch.from_numpy(data['gt_action'])
        gt_action1 = torch.from_numpy(data['gt_action1'])

        num_scene = len(self.num_agent)

        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        self.ractions = []
        self.gt_action = []
        self.gt_action1 = []
        self.Type = []
        self.CrossType = []
        self.Violation = []
        self.PI = []
        for i in range(num_scene):
            self.v_obs.append(v_obs[i, :, :self.num_agent[i], :])
            self.A_obs.append(A_obs[i, :, :self.num_agent[i], :self.num_agent[i], :])
            self.v_pred.append(v_pred[i, :, :self.num_agent[i], :])
            self.ractions.append(ractions[i, :, :self.num_agent[i]])
            self.gt_action.append(gt_action[i, :self.num_agent[i]])
            self.gt_action1.append(gt_action1[i, :self.num_agent[i]])
            self.Type.append(self.v_obs[-1][0, :, -1])
            self.CrossType.append(self.v_pred[-1][0, :, -2])
            self.Violation.append(self.v_pred[-1][0, :, -1])
            vio = self.Violation[-1]
            pi = torch.zeros(vio.shape[0], 2)
            pi[vio == 0, 0] = 1  # No Violation
            pi[vio >= 1, 1] = 1  # Violation
            self.PI.append(pi)

    def __getitem__(self, idx):
        start, end = self.se[idx]
        out = [
            self.obs_traj[start:end, :],
            self.v_obs[idx], self.A_obs[idx],
            self.v_pred[idx],
            self.Type[idx],
            self.CrossType[idx],
            self.Violation[idx],
            self.PI[idx],
            self.ractions[idx],
            self.gt_action[idx],
            self.gt_action1[idx]
        ]

        return out

    def __len__(self):
        return len(self.num_agent)
