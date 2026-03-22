import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512)):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        self.activation = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.activation.append(nn.PReLU())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation[i](x)
                x = nn.Dropout(0.9)(x)
        return x


class multidirected_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, ln_spa):
        super(multidirected_gcn, self).__init__()
        self.ln_spa = ln_spa

        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))

        self.mlp_spa = nn.ModuleList()
        self.mlp_spa.append(nn.Linear(3, 8))
        for j in range(self.ln_spa - 2):
            self.mlp_spa.append(nn.Linear(8, 8))
        self.mlp_spa.append(nn.Linear(8, 3))
        self.pool = nn.MaxPool2d((1, 4))

        self.mlp_spa1 = nn.ModuleList()
        self.mlp_spa1.append(nn.Linear(1, 8))
        for j in range(self.ln_spa - 2):
            self.mlp_spa1.append(nn.Linear(8, 8))
        self.mlp_spa1.append(nn.Linear(8, 1))

        self.ac_spa = nn.ModuleList()
        for j in range(self.ln_spa):
            self.ac_spa.append(nn.Tanh())

    def forward(self, x, A):
        A = A.view(A.shape[1], A.shape[0], A.shape[2], A.shape[3])
        A123 = A[:, :, :, :3]
        A4 = A[:, :, :, 3:]
        for i in range(self.ln_spa):
            A123 = (self.ac_spa[i](self.mlp_spa[i](A123)))
            A4 = (self.ac_spa[i](self.mlp_spa1[i](A4)))
        A = torch.cat((A123, A4), -1)
        A = self.pool(A)
        A = A.view(A.shape[3], A.shape[2], A.shape[1], A.shape[0])
        x = torch.einsum('fvdv, nctv->fctv', (A, x))
        x = self.gc(x)
        return x


class gcn(nn.Module):

    def __init__(self, in_channels, out_channels, n_layer, ln_spa):
        super(gcn, self).__init__()

        self.n_layer = n_layer

        self.gcn = nn.ModuleList()
        self.gcn.append(multidirected_gcn(in_channels, out_channels, ln_spa))
        for j in range(self.n_layer - 1):
            self.gcn.append(multidirected_gcn(out_channels, out_channels, ln_spa))

        self.prelus = nn.ModuleList()
        for j in range(self.n_layer):
            self.prelus.append(nn.PReLU())

        self.tcn = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x, A):
        tcn = self.tcn(x)
        x = self.gcn[0](x, A)
        x = x + tcn
        x = self.prelus[0](x)
        for i in range(1, self.n_layer):
            x = self.prelus[i](self.gcn[i](x, A) + x)
        return x


class ddgc(nn.Module):
    def __init__(self, n_spa, n_cnn, n_gcn, input, output, obs_len,
                 pred_len):
        super(ddgc, self).__init__()

        self.pred_len = pred_len

        if n_spa < 1:
            ln_spa = 1
        else:
            ln_spa = n_spa

        if n_cnn < 3:
            self.n_cnn = 2
        else:
            self.n_cnn = n_cnn

        if n_gcn < 1:
            self.n_gcn = 1
        else:
            self.n_gcn = n_gcn

        print('Number of Layers of MLP for Space: ', ln_spa)
        print('Number of Layers of GNN: ', self.n_gcn)
        print('Number of Layers of CNN for Multimodal: ', self.n_cnn)

        self.gcn = gcn(input, output, n_gcn, ln_spa)

        # Pedestrian
        self.cnn_pe = nn.ModuleList()
        self.cnn_pe.append(nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0)))
        for j in range(self.n_cnn):
            self.cnn_pe.append(nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)))
        self.cnn_r_pe = nn.Conv2d(obs_len, pred_len, kernel_size=1)
        self.prelus_pe = nn.ModuleList()
        for j in range(self.n_cnn):
            self.prelus_pe.append(nn.PReLU())

        # left
        self.cnn_left = nn.ModuleList()
        self.cnn_left.append(nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0)))
        for j in range(self.n_cnn):
            self.cnn_left.append(nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)))
        self.cnn_r_left = nn.Conv2d(obs_len, pred_len, kernel_size=1)
        self.prelus_left = nn.ModuleList()
        for j in range(self.n_cnn):
            self.prelus_left.append(nn.PReLU())

        # right
        self.cnn_right = nn.ModuleList()
        self.cnn_right.append(nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0)))
        for j in range(self.n_cnn):
            self.cnn_right.append(nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)))
        self.cnn_r_right = nn.Conv2d(obs_len, pred_len, kernel_size=1)
        self.prelus_right = nn.ModuleList()
        for j in range(self.n_cnn):
            self.prelus_right.append(nn.PReLU())

        # stop
        self.cnn_stop = nn.ModuleList()
        self.cnn_stop.append(nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0)))
        for j in range(self.n_cnn):
            self.cnn_stop.append(nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)))
        self.cnn_r_stop = nn.Conv2d(obs_len, pred_len, kernel_size=1)
        self.prelus_stop = nn.ModuleList()
        for j in range(self.n_cnn):
            self.prelus_stop.append(nn.PReLU())

        # straight
        self.cnn_straight = nn.ModuleList()
        self.cnn_straight.append(nn.Conv2d(obs_len, pred_len, (3, 1), padding=(1, 0)))
        for j in range(self.n_cnn):
            self.cnn_straight.append(nn.Conv2d(pred_len, pred_len, (3, 1), padding=(1, 0)))
        self.cnn_r_straight = nn.Conv2d(obs_len, pred_len, kernel_size=1)
        self.prelus_straight = nn.ModuleList()
        for j in range(self.n_cnn):
            self.prelus_straight.append(nn.PReLU())

        self.pi = MLP(16 * 5, 4, hidden_size=(1024, 1024))
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)

        self.gru_past = nn.GRU(obs_len * 2, 16, 1)
        self.gru_left = nn.GRU(obs_len * 2, 16, 1)
        self.gru_right = nn.GRU(obs_len * 2, 16, 1)
        self.gru_straight = nn.GRU(obs_len * 2, 16, 1)
        self.gru_stop = nn.GRU(obs_len * 2, 16, 1)
        self.att = nn.MultiheadAttention(16, 1)

    def forward(self, v, a, act):

        type = v[:, -1, 0, :]

        v = v.contiguous()
        a = a.contiguous()
        v = self.gcn(v, a)

        act = act.reshape(act.shape[0], act.shape[-2], act.shape[-1], -1)
        past = act[:, :, 0, :]
        left = act[:, :, 3, :]
        right = act[:, :, 4, :]
        stop = act[:, :, 1, :]
        straight = act[:, :, 2, :]

        _, past = self.gru_past(past)
        _, left = self.gru_left(left)
        _, right = self.gru_right(right)
        _, straight = self.gru_straight(straight)
        _, stop = self.gru_stop(stop)

        h1, _ = self.att(past, left, past)
        h2, _ = self.att(past, right, past)
        h3, _ = self.att(past, straight, past)
        h4, _ = self.att(past, stop, past)
        p_ = self.log_softmax(self.pi(torch.cat((past, h1, h2, h3, h4), -1)))

        p = torch.exp(p_)
        index = torch.argmax(p, -1)

        index[type == 0] = -1
        index2 = index == -1
        index0 = index == 0
        index1 = index == 1
        index3 = index == 2
        index4 = index == 3
        #[stop, right, straight, left]

        v0 = v[:, :, :, index0[0]]  # [batch, gcn_feats, steps, agents]
        v1 = v[:, :, :, index1[0]]
        v2 = v[:, :, :, index2[0]]
        v3 = v[:, :, :, index3[0]]
        v4 = v[:, :, :, index4[0]]
        # print(v0.shape, v1.shape, v2.shape)

        v = torch.zeros((v.shape[0], v.shape[1], self.pred_len, v.shape[3])).cuda()
        if v0.shape[3] != 0:  # No Violation
            # decoder
            v0 = v0.view(v0.shape[0], v0.shape[2], v0.shape[1], v0.shape[3])  # [1, steps, pdf_params, agents]
            v0 = self.prelus_left[0](self.cnn_left[0](v0) + self.cnn_r_left(v0))
            for k in range(1, self.n_cnn):
                v0 = self.prelus_left[k](self.cnn_left[k](v0) + v0)
            v0 = self.cnn_left[self.n_cnn](v0)
            v0 = v0.view(v0.shape[0], v0.shape[2], v0.shape[1], v0.shape[3])  # [1, pdf_params, steps, agents]
            v[:, :, :, index0[0]] = v0

        if v1.shape[3] != 0:  #
            # decoder
            v1 = v1.view(v1.shape[0], v1.shape[2], v1.shape[1], v1.shape[3])  # [1, steps, pdf_params, agents]
            v1 = self.prelus_right[0](self.cnn_right[0](v1) + self.cnn_r_right(v1))
            for k in range(1, self.n_cnn):
                v1 = self.prelus_right[k](self.cnn_right[k](v1) + v1)
            v1 = self.cnn_right[self.n_cnn](v1)
            v1 = v1.view(v1.shape[0], v1.shape[2], v1.shape[1], v1.shape[3])
            v[:, :, :, index1[0]] = v1

        if v2.shape[3] != 0:
            # decoder
            v2 = v2.view(v2.shape[0], v2.shape[2], v2.shape[1], v2.shape[3])  # [1, steps, pdf_params, agents]
            v2 = self.prelus_pe[0](self.cnn_left[0](v2) + self.cnn_r_pe(v2))
            for k in range(1, self.n_cnn):
                v2 = self.prelus_pe[k](self.cnn_pe[k](v2) + v2)
            v2 = self.cnn_pe[self.n_cnn](v2)
            v2 = v2.view(v2.shape[0], v2.shape[2], v2.shape[1], v2.shape[3])
            v[:, :, :, index2[0]] = v2

        if v3.shape[3] != 0:  #
            # decoder
            v3 = v3.view(v3.shape[0], v3.shape[2], v3.shape[1], v3.shape[3])  # [1, steps, pdf_params, agents]
            v3 = self.prelus_straight[0](self.cnn_straight[0](v3) + self.cnn_r_straight(v3))
            for k in range(1, self.n_cnn):
                v3 = self.prelus_straight[k](self.cnn_straight[k](v3) + v3)
            v3 = self.cnn_straight[self.n_cnn](v3)
            v3 = v3.view(v3.shape[0], v3.shape[2], v3.shape[1], v3.shape[3])
            v[:, :, :, index3[0]] = v3

        if v4.shape[3] != 0:  #
            # decoder
            v4 = v4.view(v4.shape[0], v4.shape[2], v4.shape[1], v4.shape[3])  # [1, steps, pdf_params, agents]
            v4 = self.prelus_stop[0](self.cnn_stop[0](v4) + self.cnn_r_stop(v4))
            for k in range(1, self.n_cnn):
                v4 = self.prelus_stop[k](self.cnn_stop[k](v4) + v4)
            v4 = self.cnn_stop[self.n_cnn](v4)
            v4 = v4.view(v4.shape[0], v4.shape[2], v4.shape[1], v4.shape[3])
            v[:, :, :, index4[0]] = v4

        # v.shape [batch, params, steps, agents]
        return v, p_, index

