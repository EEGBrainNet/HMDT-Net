import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from frequency import FFT_b_1,FFT_b_2
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
def non_overlapping_sliding_window_3d(tensor1, tensor2):
    windows1,windows2 = [],[]
    for i in range(3):
        if i==1:
            window1 = tensor1[:, :, 1000: 2000]
            window2 = tensor2[:, :, 1000: 2000]

        elif i==2:
            window1 = tensor1[:, :, 2000: 3000]
            window2 = tensor2[:, :, 2000: 3000]

        else:
            window1 = tensor1[:, :, 0: 1000]
            window2 = tensor2[:, :, 0: 1000]

        windows1.append(window1)
        windows2.append(window2)

    windows1 = torch.stack(windows1, dim=1)
    windows2 = torch.stack(windows2, dim=1)
    return windows1, windows2

class EEGFeatureExtractor(nn.Module):
    def __init__(self,):
        super(EEGFeatureExtractor, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=2, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )
        self.lstm = nn.LSTM(66, 30, 2, batch_first=True, bidirectional=True)
        self.features6 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=2, bias=False, padding=10),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=13, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)

    def forward(self, x):
        x1 = self.features1(x)
        lstm_out, (h, c) = self.lstm(x1, None)
        x2=lstm_out
        f = FFT_b_2(x)
        x6 = self.features6(f)
        x_concat = torch.cat((x2, x6), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat

class EOGFeatureExtractor(nn.Module):
    def __init__(self, ):
        super(EOGFeatureExtractor, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )
        self.lstm = nn.LSTM(66, 30, 2, batch_first=True, bidirectional=True)
        self.features6 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, bias=False, padding=10),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=13, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)

    def forward(self, x):
        x1 = self.features1(x)
        lstm_out, (h, c) = self.lstm(x1, None)
        x2=lstm_out
        f = FFT_b_1(x)
        x6 = self.features6(f)
        x_concat = torch.cat((x2, x6), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat

def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedSelfAttention_EEG(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedSelfAttention_EEG, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(128, 128, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


class MultiHeadedSelfAttention_EOG(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedSelfAttention_EOG, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(128, 128, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(15360, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ResidualBlock(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, d_model):
        super(ResidualBlock, self).__init__()
        self.norm = LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 200)
        self.linear2 = nn.Linear(200, d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        sub = self.norm(x)
        sub = self.linear1(sub)
        sub = F.relu(sub)
        sub = self.dropout1(sub)
        sub = self.linear2(sub)
        sub = self.dropout2(sub)
        return x + sub


class croatt(nn.Module):
    def __init__(self, d_model):
        super(croatt, self).__init__()
        h = 5
        self.conv1 = CausalConv1d(128, 128, kernel_size=7, stride=1, dilation=1)
        self.conv2 = CausalConv1d(128, 128, kernel_size=7, stride=1, dilation=1)
        self.norm1 = LayerNorm(120)
        self.norm2 = LayerNorm(120)
        self.mhsa_EEG = MultiHeadedSelfAttention_EEG(h, 120)
        self.mhsa_EOG = MultiHeadedSelfAttention_EOG(h, 120)
        self.rb1 = ResidualBlock(d_model=120)
        self.rb2 = ResidualBlock(d_model=120)

    def forward(self, x, x_p):
        Query_EEG = self.conv1(x)
        Query_EOG = self.conv2(x_p)
        attnq_EEG = self.mhsa_EEG(self.norm1(Query_EEG), x, x)
        attnq_EEG = x * attnq_EEG
        attnq_EOG = self.mhsa_EOG(self.norm2(Query_EOG), x_p, x_p)
        attnq_EOG = x * attnq_EOG
        Query_EEG = self.conv1(attnq_EEG)
        Query_EOG = self.conv2(attnq_EOG)
        attnq_EEG = self.mhsa_EEG(self.norm1(Query_EEG), attnq_EOG, attnq_EOG)
        attnq_EOG = self.mhsa_EOG(self.norm2(Query_EOG), attnq_EEG, attnq_EEG)
        attnq_EEG = attnq_EEG + Query_EEG
        attnq_EOG = attnq_EOG + Query_EOG

        attnq_EEG = self.rb1(attnq_EEG)
        attnq_EOG = self.rb2(attnq_EOG)

        out = attnq_EEG + attnq_EOG

        return out

class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd=0.01):
        ctx.lmbd = lmbd
        return x.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.lmbd * grad_input.neg(), None


def get_alpha(cur_step, total_step):
    p = cur_step / total_step
    return 2 / (1 + np.exp(-10 * p)) - 1

class HMDTF(nn.Module):
    def __init__(self):
        super(HMDTF, self).__init__()

        d_model = 120
        num_classes = 5
        hidden_layer = 1024
        domain_classes = 2
        afr_reduced_cnn_size1=30
        self.RE = nn.Sequential(
            nn.Conv1d(128, 30, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True)
        )
        self.ca = croatt(d_model)
        self.fc = nn.Linear(46080, num_classes)
        self.EEGfe=EEGFeatureExtractor(afr_reduced_cnn_size1)
        self.EOGfe = EOGFeatureExtractor(afr_reduced_cnn_size1)
        self.Domain_classifier = nn.Sequential(
            nn.Linear(46080, hidden_layer),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_layer),
            nn.Linear(hidden_layer,domain_classes)
        )
        self.T_l = nn.Sequential(
            nn.Linear(15360, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
        )
        self.p_l = nn.Sequential(
            nn.Linear(15360, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
        )
        self.alpha = 0.01
    def forward(self,x):

            x = x.float()
            x1 = x
            x_pzcz = x1[:, 0, :]
            x_pzoz = x1[:, 1, :]
            x_eog = x1[:, 2, :]

            x_pzcz = x_pzcz.unsqueeze(1)
            x_pzoz = x_pzoz.unsqueeze(1)
            x_eog = x_eog.unsqueeze(1)

            x = torch.cat((x_pzcz, x_pzoz), dim=1)
            x_p = x_eog
            window_size = 1000
            windows1, windows2 = non_overlapping_sliding_window_3d(x,x_p, window_size)

            xencoder1 = []
            tl = []
            pl = []

            for i in range(0, 3):
                x = windows1[:, i, :, :]
                x_p = windows2[:, i, :, :]

                x = self.EEGfe(x)
                x_p = self.EOGfe(x_p)

                T_l = x.contiguous().view(x.shape[0], -1)
                T_l = self.T_l(T_l)
                p_l = x_p.contiguous().view(x_p.shape[0], -1)
                p_l = self.p_l(p_l)

                x_encode= self.ca(x, x_p)

                xencoder1.append(x_encode)
                tl.append(T_l)
                pl.append(p_l)

            xencoder2 = torch.stack(xencoder1, dim=1)
            tl1 = torch.stack(tl, dim=1)
            pl1 = torch.stack(pl, dim=1)

            x_encode = xencoder2.contiguous().view(xencoder2.shape[0], -1)
            rev_fea = GRLayer.apply(x_encode, self.alpha)
            final_output = self.fc(x_encode)
            domain_class = self.Domain_classifier(rev_fea)

            tl1 = torch.mean(tl1, dim=1)
            pl1 = torch.mean(pl1, dim=1)

            return final_output, tl1, pl1, domain_class, x_encode



