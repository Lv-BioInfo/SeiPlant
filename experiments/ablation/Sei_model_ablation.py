"""
Sei architecture
"""
import numpy as np
import torch
import torch.nn as nn



import numpy as np
from scipy.interpolate import splev


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()



class BSplineTransformation(nn.Module):

    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        device = input.device  # 获取输入张量的设备

        # 初始化 _spline_tr，且只在其为 None 的情况下进行初始化
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]  # 获取输入张量的最后一个维度的大小
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)

            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim

        # 确保 _spline_tr 在与输入张量相同的设备上
        self._spline_tr = self._spline_tr.to(device)

        # 进行矩阵乘法操作
        return torch.matmul(input, self._spline_tr)



class BSplineConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, degrees_of_freedom, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, log=False, scaled = True):
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled

        self.spline = nn.Conv1d(1, degrees_of_freedom, kernel_size, stride, padding, dilation,
            bias=False)
        self.spline.weight = spline_factory(kernel_size, self._df, log=log).view(self._df, 1, kernel_size)
        if scaled:
            self.spline.weight = self.spline.weight / kernel_size
        self.spline.weight = nn.Parameter(self.spline.weight)
        self.spline.weight.requires_grad = False
        self.conv1d = nn.Conv1d(in_channels * degrees_of_freedom, out_channels, 1,
            groups = groups, bias=bias)

    def forward(self, input):
        batch_size, n_channels, length = input.size()
        spline_out = self.spline(input.view(batch_size * n_channels,1,length))
        conv1d_out = self.conv1d(spline_out.view(batch_size, n_channels * self._df,  length))
        return conv1d_out

# class Sei(nn.Module):
#     def __init__(self, sequence_length=4096, n_genomic_features=21907,
#                  remove_dilated=False, remove_residual=False, remove_spline=False):
#         super(Sei, self).__init__()
#         self.n_genomic_features = n_genomic_features  # 存起来
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.remove_dilated = remove_dilated
#         self.remove_residual = remove_residual
#         self.remove_spline = remove_spline

#         self.lconv1 = nn.Sequential(
#             nn.Conv1d(4, 480, kernel_size=9, padding=4),
#             nn.Conv1d(480, 480, kernel_size=9, padding=4))

#         self.conv1 = nn.Sequential(
#             nn.Conv1d(480, 480, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(480, 480, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True))

#         self.lconv2 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=4, stride=4),
#             nn.Dropout(p=0.2),
#             nn.Conv1d(480, 640, kernel_size=9, padding=4),
#             nn.Conv1d(640, 640, kernel_size=9, padding=4))

#         self.conv2 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Conv1d(640, 640, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(640, 640, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True))

#         self.lconv3 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=4, stride=4),
#             nn.Dropout(p=0.2),
#             nn.Conv1d(640, 960, kernel_size=9, padding=4),
#             nn.Conv1d(960, 960, kernel_size=9, padding=4))

#         self.conv3 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Conv1d(960, 960, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(960, 960, kernel_size=9, padding=4),
#             nn.ReLU(inplace=True))

#         if not self.remove_dilated:
#             self.dconv1 = nn.Sequential(
#                 nn.Dropout(p=0.10),
#                 nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
#                 nn.ReLU(inplace=True))
#             self.dconv2 = nn.Sequential(
#                 nn.Dropout(p=0.10),
#                 nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
#                 nn.ReLU(inplace=True))
#             self.dconv3 = nn.Sequential(
#                 nn.Dropout(p=0.10),
#                 nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
#                 nn.ReLU(inplace=True))
#             self.dconv4 = nn.Sequential(
#                 nn.Dropout(p=0.10),
#                 nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
#                 nn.ReLU(inplace=True))
#             self.dconv5 = nn.Sequential(
#                 nn.Dropout(p=0.10),
#                 nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
#                 nn.ReLU(inplace=True))

#         # Spline 层开关
#         self._spline_df = int(128 / 8)
#         if self.remove_spline:
#             self.spline_tr = nn.Identity()
#             self.classifier_in_dim = None  # 训练时自动计算
#         else:
#             self.spline_tr = nn.Sequential(
#                 nn.Dropout(p=0.5),
#                 BSplineTransformation(self._spline_df, scaled=False))
#             self.classifier_in_dim = 960 * self._spline_df  # 固定维度

#             # 分类器先定义成占位，forward 时根据输入维度重新初始化（只初始化一次）
#         self.classifier = None

#     def _build_classifier(self, in_dim, n_genomic_features):
#         """根据输入维度动态初始化分类器"""
#         self.classifier = nn.Sequential(
#             nn.Linear(in_dim, n_genomic_features),
#             nn.ReLU(inplace=True),
#             nn.Linear(n_genomic_features, n_genomic_features),
#             nn.Sigmoid()
#         ).to(self.device)

#     def forward(self, x):
#         lout1 = self.lconv1(x)
#         out1 = self.conv1(lout1)

#         lout2 = self.lconv2(out1 if self.remove_residual else (out1 + lout1))
#         out2 = self.conv2(lout2)

#         lout3 = self.lconv3(out2 if self.remove_residual else (out2 + lout2))
#         out3 = self.conv3(lout3)

#         if self.remove_dilated:
#             out = out3 if self.remove_residual else (out3 + lout3)
#         else:
#             dconv_out1 = self.dconv1(out3 if self.remove_residual else (out3 + lout3))
#             cat_out1 = out3 + dconv_out1
#             dconv_out2 = self.dconv2(cat_out1)
#             cat_out2 = cat_out1 + dconv_out2
#             dconv_out3 = self.dconv3(cat_out2)
#             cat_out3 = cat_out2 + dconv_out3
#             dconv_out4 = self.dconv4(cat_out3)
#             cat_out4 = cat_out3 + dconv_out4
#             dconv_out5 = self.dconv5(cat_out4)
#             out = cat_out4 + dconv_out5

#         spline_out = self.spline_tr(out)
#         # 动态计算展平维度
#         batch_size = spline_out.size(0)
#         flatten_dim = spline_out.size(1) * spline_out.size(2)
#         reshape_out = spline_out.view(batch_size, flatten_dim)

#         # 初始化分类器（只在第一次 forward 时执行）
#         if self.classifier is None:
#             self._build_classifier(flatten_dim, n_genomic_features=self.n_genomic_features)

#         predict = self.classifier(reshape_out)
#         return predict

class Sei(nn.Module):
    def __init__(self, sequence_length=4096, n_genomic_features=21907,
                 remove_dilated=False, remove_residual=False, remove_spline=False):
        super(Sei, self).__init__()

        self.remove_dilated = remove_dilated
        self.remove_residual = remove_residual
        self.remove_spline = remove_spline

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        if not self.remove_dilated:
            self.dconv1 = nn.Sequential(
                nn.Dropout(p=0.10),
                nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
                nn.ReLU(inplace=True))
            self.dconv2 = nn.Sequential(
                nn.Dropout(p=0.10),
                nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
                nn.ReLU(inplace=True))
            self.dconv3 = nn.Sequential(
                nn.Dropout(p=0.10),
                nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
                nn.ReLU(inplace=True))
            self.dconv4 = nn.Sequential(
                nn.Dropout(p=0.10),
                nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
                nn.ReLU(inplace=True))
            self.dconv5 = nn.Sequential(
                nn.Dropout(p=0.10),
                nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
                nn.ReLU(inplace=True))

        # Spline 层开关
        self._spline_df = int(128 / 8)
        if self.remove_spline:
            self.spline_tr = nn.Identity()  # 直接跳过
        else:
            self.spline_tr = nn.Sequential(
                nn.Dropout(p=0.5),
                BSplineTransformation(self._spline_df, scaled=False))

        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())


    def forward(self, x):
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 if self.remove_residual else (out1 + lout1))
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 if self.remove_residual else (out2 + lout2))
        out3 = self.conv3(lout3)

        if self.remove_dilated:
            out = out3 if self.remove_residual else (out3 + lout3)
        else:
            dconv_out1 = self.dconv1(out3 if self.remove_residual else (out3 + lout3))
            cat_out1 = out3 + dconv_out1
            dconv_out2 = self.dconv2(cat_out1)
            cat_out2 = cat_out1 + dconv_out2
            dconv_out3 = self.dconv3(cat_out2)
            cat_out3 = cat_out2 + dconv_out3
            dconv_out4 = self.dconv4(cat_out3)
            cat_out4 = cat_out3 + dconv_out4
            dconv_out5 = self.dconv5(cat_out4)
            out = cat_out4 + dconv_out5

        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        predict = self.classifier(reshape_out)
        return predict

class Sei_NoSpline(nn.Module):
    def __init__(self, sequence_length=4096, n_genomic_features=21907,
                 remove_dilated=False, remove_residual=False):
        super(Sei_NoSpline, self).__init__()

        self.remove_dilated = remove_dilated
        self.remove_residual = remove_residual
        self.n_genomic_features = n_genomic_features

        # ========= 卷积块 =========
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        # ========= 空洞卷积 =========
        if not self.remove_dilated:
            self.dconv1 = nn.Sequential(nn.Dropout(0.10), nn.Conv1d(960, 960, 5, dilation=2, padding=4), nn.ReLU())
            self.dconv2 = nn.Sequential(nn.Dropout(0.10), nn.Conv1d(960, 960, 5, dilation=4, padding=8), nn.ReLU())
            self.dconv3 = nn.Sequential(nn.Dropout(0.10), nn.Conv1d(960, 960, 5, dilation=8, padding=16), nn.ReLU())
            self.dconv4 = nn.Sequential(nn.Dropout(0.10), nn.Conv1d(960, 960, 5, dilation=16, padding=32), nn.ReLU())
            self.dconv5 = nn.Sequential(nn.Dropout(0.10), nn.Conv1d(960, 960, 5, dilation=25, padding=50), nn.ReLU())

        # ⚠️ 分类器先不定义，forward 时根据输入动态创建
        self.classifier = None

    def forward(self, x):
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 if self.remove_residual else (out1 + lout1))
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 if self.remove_residual else (out2 + lout2))
        out3 = self.conv3(lout3)

        if self.remove_dilated:
            out = out3 if self.remove_residual else (out3 + lout3)
        else:
            dconv_out1 = self.dconv1(out3 if self.remove_residual else (out3 + lout3))
            cat_out1 = out3 + dconv_out1
            dconv_out2 = self.dconv2(cat_out1)
            cat_out2 = cat_out1 + dconv_out2
            dconv_out3 = self.dconv3(cat_out2)
            cat_out3 = cat_out2 + dconv_out3
            dconv_out4 = self.dconv4(cat_out3)
            cat_out4 = cat_out3 + dconv_out4
            dconv_out5 = self.dconv5(cat_out4)
            out = cat_out4 + dconv_out5

        # ⚠️ 直接展平卷积输出
        batch_size = out.size(0)
        flatten_dim = out.size(1) * out.size(2)
        reshape_out = out.view(batch_size, flatten_dim)

        # ⚠️ 第一次 forward 时动态创建分类器
        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(flatten_dim, self.n_genomic_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_genomic_features, self.n_genomic_features),
                nn.Sigmoid()
            ).to(out.device)

        return self.classifier(reshape_out)


def criterion():
    """
    The criterion the model aims to minimize.
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-7, "momentum": 0.9})