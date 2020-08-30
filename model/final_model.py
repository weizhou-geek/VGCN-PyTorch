import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import Parameter
import math
from torchvision import models

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()

        self.gc1 = GraphConvolution(512, 256)
        self.bn1 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc2 = GraphConvolution(256, 128)
        self.bn2 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc3 = GraphConvolution(128, 64)
        self.bn3 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc4 = GraphConvolution(64, 32)
        self.bn4 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc5 = GraphConvolution(32, 1)
        self.relu = nn.Softplus()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def norm_adj(self, matrix):
        D = torch.diag_embed(matrix.sum(2))
        D = D ** 0.5
        D = D.inverse()
        # D(-1/2) * A * D(-1/2)
        normal = D.bmm(matrix).bmm(D)
        return normal.detach()

    def forward(self, feature, A):
        adj = self.norm_adj(A)
        gc1 = self.gc1(feature, adj)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc2 = self.gc2(gc1, adj)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc3 = self.gc3(gc2, adj)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)
        gc4 = self.gc4(gc3, adj)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)
        gc5 = self.gc5(gc4, adj)
        gc5 = self.relu(gc5)
        return gc5

class OIQANet(nn.Module):
    def __init__(self, model):
        super(OIQANet, self).__init__()

        self.resnet = nn.Sequential(*list(model.children())[:-2])
        self.maxpool = nn.MaxPool2d(8)
        self.GCN = GCNNet()
        self.fc = nn.Linear(20, 1)

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_build(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True)
        return distortion

    def forward(self, x, label, A, requires_loss):
        batch_size = x.size(0)
        y = x.view(-1, 3, 256, 256)
        all_feature = self.resnet(y)
        all_feature = self.maxpool(all_feature)
        feature = all_feature.view(batch_size, 20, -1, 1, 1)
        feature = feature.squeeze(3)
        feature = feature.squeeze(3)

        gc5 = self.GCN(feature, A)
        fc_in = gc5.view(gc5.size()[0], -1)
        score = torch.mean(fc_in, dim=1).unsqueeze(1)

        if requires_loss:
            return score, label, self.loss_build(score, label)
        else:
            return score


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39

        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
#        return X
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
        return X

class DBCNN(torch.nn.Module):

    def __init__(self, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        # self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = models.vgg16(pretrained=False).features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
        scnn = SCNN()
        # scnn = torch.nn.DataParallel(scnn).cuda()
        # scnn.load_state_dict({k.replace('module.', ""): v for k, v in torch.load(scnn_root).items()})
        # scnn.load_state_dict(torch.load(scnn_root))
        # print('load scnn model!')
        self.features2 = scnn.features

        # Linear classifier.
        self.fc = torch.nn.Linear(512 * 128, 1)

        if options['fc'] == True:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def loss_build(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True)
        return distortion

    def forward(self, X, label, requires_loss):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128

        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2, (H, W))

        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512 * 128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 1)
        if requires_loss:
            return X, label, self.loss_build(X, label)
        else:
            return X

class VGCN(torch.nn.Module):

    def __init__(self, root1, root2):
        super(VGCN, self).__init__()

        res_net = models.resnet18(pretrained=False)
        self.OIQA_branch = OIQANet(res_net)
        if root1:
            pretrained_dict1 = torch.load(root1)
            oiqa_dict = self.OIQA_branch.state_dict()
            pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in oiqa_dict}
            oiqa_dict.update(pretrained_dict1)
            self.OIQA_branch.load_state_dict(oiqa_dict)
            print('OIQA_branch model load!')

        options = {'fc': []}
        options['fc'] = True
        self.DBCNN_branch = DBCNN(options=options)
        if root2:
            pretrained_dict2 = torch.load(root2)
            dbcnn_dict = self.DBCNN_branch.state_dict()
            pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in dbcnn_dict}
            dbcnn_dict.update(pretrained_dict2)
            self.DBCNN_branch.load_state_dict(dbcnn_dict)
            print('DBCNN_branch model load!')

        self.fc = nn.Linear(2, 1)

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_build(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True)
        return distortion

    def forward(self, fov, whole, label, A, requires_loss):

        score1 = self.OIQA_branch(fov, label, A, requires_loss=False)
        score2 = self.DBCNN_branch(whole, label, requires_loss=False)
        score_fuse = torch.cat((score1, score2), dim=1)
        score = self.fc(score_fuse)

        if requires_loss:
            return score, label, self.loss_build(score, label)
        else:
            return score, label

