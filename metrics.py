import math

import numpy as np
import torch


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_, max_nodes=88):
    if seq_.shape[2] != 2:
        seq_ = seq_[:, :, :-1, :]
    seq_ = seq_.squeeze(0)
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result


def bivariate_loss_(V_pred, V_trgt, num):
    for i in range(num):
        normx = V_trgt[:, :, 0] - V_pred[:, :, i * 6 + 0]
        normy = V_trgt[:, :, 1] - V_pred[:, :, i * 6 + 1]

        sx = torch.exp(V_pred[:, :, i * 6 + 2])  # sx
        sy = torch.exp(V_pred[:, :, i * 6 + 3])  # sy
        corr = torch.tanh(V_pred[:, :, i * 6 + 4])  # corr

        sxsy = sx * sy

        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
        negRho = 1 - corr ** 2

        # Numerator
        result = torch.exp(-z / (2 * negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        if i == 0:
            # Final PDF calculation
            r = result / denom * torch.tanh(V_pred[:, :, i * 6 + 5])
        else:
            r = r + result / denom * torch.tanh(V_pred[:, :, i * 6 + 5])

    # Numerical stability
    epsilon = 1e-20

    r = -torch.log(torch.clamp(r, min=epsilon))
    r = torch.mean(r)

    return r


def bivariate_loss__(V_pred, V_trgt, num):
    r, w = torch.zeros((num, V_pred.shape[0], V_pred.shape[1])).cuda(), torch.zeros(
        (num, V_pred.shape[0], V_pred.shape[1])).cuda()
    for i in range(num):
        normx = V_trgt[:, :, 0] - V_pred[:, :, i * 6 + 0]
        normy = V_trgt[:, :, 1] - V_pred[:, :, i * 6 + 1]

        sx = torch.exp(V_pred[:, :, i * 6 + 2])  # sx
        sy = torch.exp(V_pred[:, :, i * 6 + 3])  # sy
        corr = torch.tanh(V_pred[:, :, i * 6 + 4])  # corr

        sxsy = sx * sy

        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
        negRho = 1 - corr ** 2

        # Numerator
        result = torch.exp(-z / (2 * negRho))
        # Normalization factor
        denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

        r[i, :, :] = result / denom
        w[i, :, :] = (torch.tanh(V_pred[:, :, i * 6 + 5]) + 1) / 2

        if i == 0:
            sum = w[i, :, :]
        else:
            sum = sum + w[i, :, :]

    for i in range(num):
        if i == 0:
            result = r[i, :, :] * (w[i, :, :] / sum)
        else:
            result = result * r[i, :, :] * (w[i, :, :] / sum)

    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result


def loss_(V_pred, V_trgt):
    normx = abs(V_trgt[:, :, 0] - V_pred[:, :, 0])
    normy = abs(V_trgt[:, :, 1] - V_pred[:, :, 1])

    result = torch.mean(normx) + torch.mean(normy)

    return result


def bivariate_loss_mix(V_pred, V_trgt):
    normx1 = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy1 = V_trgt[:, :, 1] - V_pred[:, :, 1]

    xx = torch.exp(V_pred[:, :, 2])  # sx
    yy = torch.exp(V_pred[:, :, 3])  # sy

    q = 1 / math.pi

    x = q * xx / (normx1 ** 2 + xx ** 2)
    y = q * yy / (normy1 ** 2 + yy ** 2)

    normx2 = V_trgt[:, :, 0] - V_pred[:, :, 4]
    normy2 = V_trgt[:, :, 1] - V_pred[:, :, 5]

    sx = torch.exp(V_pred[:, :, 6])  # sx
    sy = torch.exp(V_pred[:, :, 7])  # sy
    corr = torch.tanh(V_pred[:, :, 8])  # corr

    sxsy = sx * sy

    z = (normx2 / sx) ** 2 + (normy2 / sy) ** 2 - 2 * ((corr * normx2 * normy2) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    z = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    z = z / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(z * (x * y), min=epsilon))
    result = torch.mean(result)

    return result


def bivariate_loss_mix1(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    xx = torch.exp(V_pred[:, :, 2])  # sx
    yy = torch.exp(V_pred[:, :, 3])  # sy

    q = 1 / math.pi

    x1 = q * xx / (normx ** 2 + xx ** 2)
    y1 = q * yy / (normy ** 2 + yy ** 2)

    sx = torch.exp(V_pred[:, :, 4])  # sx
    sy = torch.exp(V_pred[:, :, 5])  # sy

    x2 = torch.exp(-(normx ** 2) / (2 * sx * sx)) / (math.sqrt(2 * math.pi) * sx)
    y2 = torch.exp(-(normy ** 2) / (2 * sy * sy)) / (math.sqrt(2 * math.pi) * sy)

    # Numerical stability
    epsilon = 1e-200

    result = -torch.log(torch.clamp((x1 + y1) + (x2 + y2), min=epsilon))
    result = torch.mean(result)

    return result


def bivariate_loss_mix2(V_pred, V_trgt):
    normx1 = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy1 = V_trgt[:, :, 1] - V_pred[:, :, 1]

    xx = torch.exp(V_pred[:, :, 2])  # sx
    yy = torch.exp(V_pred[:, :, 3])  # sy

    q = 1 / math.pi

    x1 = q * xx / (normx1 ** 2 + xx ** 2)
    y1 = q * yy / (normy1 ** 2 + yy ** 2)

    normx2 = V_trgt[:, :, 0] - V_pred[:, :, 4]
    normy2 = V_trgt[:, :, 1] - V_pred[:, :, 5]

    sx = torch.exp(V_pred[:, :, 6])  # sx
    sy = torch.exp(V_pred[:, :, 7])  # sy

    x2 = torch.exp(-(normx2 ** 2) / (2 * sx * sx)) / (math.sqrt(2 * math.pi) * sx)
    y2 = torch.exp(-(normy2 ** 2) / (2 * sy * sy)) / (math.sqrt(2 * math.pi) * sy)

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp((x1 * y1) * (x2 * y2), min=epsilon))
    result = torch.mean(result)

    return result


def bivariate_loss_1(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy

    q = math.sqrt(2 * math.pi)

    x = torch.exp(-(normx ** 2) / (2 * sx ** 2)) / (q * sx)
    y = torch.exp(-(normy ** 2) / (2 * sy ** 2)) / (q * sy)

    result = -torch.log(torch.clamp(x * y, min=1e-20))
    result = torch.mean(result)

    return result


def bivariate_loss_2(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy

    q = 1 / math.pi

    x = q * sx / (normx ** 2 + sx ** 2)
    y = q * sy / (normy ** 2 + sy ** 2)

    result = -torch.log(torch.clamp(x * y, min=1e-10))
    result = torch.mean(result)

    return result


def bivariate_loss_3(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy

    q = 1 / math.pi

    x = q * sx / (normx ** 2 + sx ** 2)
    y = q * sy / (normy ** 2 + sy ** 2)

    result = -(torch.log(torch.clamp(x, min=1e-10)) + torch.log(torch.clamp(y, min=1e-10)))
    result = torch.mean(result)

    return result


def bivariate_loss1(V_pred, V_trgt):
    mx = (V_trgt[:, :, 0] - V_pred[:, :, 0]) ** 2
    my = (V_trgt[:, :, 1] - V_pred[:, :, 1]) ** 2

    ax = V_pred[:, :, 2] + V_trgt[:, :, 0]
    bx = V_trgt[:, :, 0] - V_pred[:, :, 3]

    ay = V_pred[:, :, 4] + V_trgt[:, :, 1]
    by = V_trgt[:, :, 1] - V_pred[:, :, 5]

    result1 = 1.0 / (ax - bx) / (ay - by)

    result2 = mx + my

    result = torch.mean(-0.1 * result1 + result2)
    return result


def student_loss(V_pred, V_trgt, low_bound):
    mx = (V_trgt[:, :, 0] - V_pred[:, :, 0])
    my = (V_trgt[:, :, 1] - V_pred[:, :, 1])

    xn = torch.exp(V_pred[:, :, 2]) + low_bound
    yn = torch.exp(V_pred[:, :, 3]) + low_bound

    xs = torch.exp(V_pred[:, :, 4])
    ys = torch.exp(V_pred[:, :, 5])

    xgam1 = torch.lgamma((xn + 1) / 2)
    xgam2 = torch.lgamma((xn) / 2) + torch.log(torch.sqrt(math.pi * xn)) + torch.log(xs)

    ygam1 = torch.lgamma((yn + 1) / 2)
    ygam2 = torch.lgamma((yn) / 2) + torch.log(torch.sqrt(math.pi * yn)) + torch.log(ys)

    x = (xgam1 - xgam2) + (-(xn + 1) / 2) * torch.log(1 + (mx / xs) * (mx / xs) / xn)
    y = (ygam1 - ygam2) + (-(yn + 1) / 2) * torch.log(1 + (my / ys) * (my / ys) / yn)

    result = torch.mean(-(x + y))
    return result
