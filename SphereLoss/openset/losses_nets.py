import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
from typing import Tuple

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m, s, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        ww = F.normalize(self.weight, p=2, dim=0)
        cos_theta = torch.mm(input, ww)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * self.s
        phi_theta = phi_theta * self.s
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

class AngleLoss(nn.Module):
    def __init__(self, gamma=0.0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 30.0
        self.lamb = self.LambdaMax

    def forward(self, input, target):
        if self.training:
            self.it += 1
            self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.5 * self.it))
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = (cos_theta.data * 0.0).bool()
        index.scatter_(1, target.data.view(-1, 1),True)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.5*self.it ))
        output = cos_theta.clone()
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class resnet(nn.Module):
    def __init__(self, num_classes, feature_dim, s, m):
        super(resnet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Linear(2048, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.angle_linear = AngleLinear(feature_dim, num_classes, s=s, m=m)
        self.feature = False

    def forward(self, x):
        x = self.backbone(x)    
        x = self.fc(x)          
        x = self.layer_norm(x)  
        x = self.dropout(x)     
        x = F.normalize(x)    
        
        if self.feature: return x
        
        out = self.angle_linear(x)
        return out
        
class sphere20a(nn.Module):
    def __init__(self,classnum=901,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.dropout_conv = nn.Dropout2d(p=0.3) # new

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc6 = AngleLinear(512,self.classnum)


    def forward(self, x):

        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = self.dropout_conv(x)

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.reshape(x.size(0), -1) # had to change
        x = self.fc5(x)
        x = self.dropout(x) 
        x = F.normalize(x)  # normalized

        if self.feature: return x

        x = self.fc6(x)
        return x

class RefinementNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 20.0)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.input_dim == self.output_dim:
            identity = x
            out = self.layers(x)
            out = out + identity
        else:
            out = self.layers(x)
        return F.normalize(out, p=2, dim=1)

class RefinementLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        
    def forward(self, embeddings, labels, temperature):
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_matrix.float()
        neg_mask = (~labels_matrix).float()
        
        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)
        
        scaled_sim_matrix = sim_matrix * temperature
        
        pos_pairs = scaled_sim_matrix * pos_mask
        pos_loss = torch.sum(F.relu(1 - pos_pairs) ** 2 * pos_mask) / (pos_mask.sum() + 1e-6)
        
        neg_pairs = scaled_sim_matrix * neg_mask
        neg_mean = torch.sum(neg_pairs * neg_mask) / (neg_mask.sum() + 1e-6)
        neg_var = torch.sum(((neg_pairs - neg_mean) ** 2) * neg_mask) / (neg_mask.sum() + 1e-6)
        
        total_loss = pos_loss + torch.abs(neg_mean) + self.beta * neg_var
        
        return total_loss, {
            'pos_loss': pos_loss.item(),
            'neg_mean': neg_mean.item(),
            'neg_var': neg_var.item(),
            'temperature': temperature.item()
        }

def convert_label_to_similarity(normed_feature: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

class ResNet18Circle(nn.Module): # custom resnet18, outputs embeddings for circleloss
    def __init__(self, feature_dim=512, dropout_p=0.5):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.feature = False

    def forward(self, x=torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)
        return x



class ResNet18SphereFace(nn.Module): # custom resnet18 with sphereface head
    def __init__(self, num_classes, feature_dim = 512, s = 64.0, m = 4, dropout_p=0.5):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        self.backbone.fc = nn.Identity()
        self.fc_embed = nn.Linear(512, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.angle_linear = AngleLinear(in_features=feature_dim, out_features=num_classes, m=m,s=s, phiflag=True)

        self.feature = False

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fc_embed(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)

        if self.feature: return x

        cos_theta, phi_theta = self.angle_linear(x)
        return cos_theta, phi_theta



class ResNet34SphereFace(nn.Module): # custom resnet18 with sphereface head
    def __init__(self, num_classes, feature_dim = 512, s = 64.0, m = 4, dropout_p=0.5):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT
        self.backbone = resnet34(weights=weights)
        self.backbone.fc = nn.Identity()

        self.fc_embed = nn.Linear(512, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        from losses_nets import AngleLinear
        self.angle_linear = AngleLinear(in_features=feature_dim, out_features=num_classes, m=m,s=s, phiflag=True)
        self.feature = False

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fc_embed(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)

        if self.feature: return x

        return self.angle_linear(x)

class ResNet34Circle(nn.Module): # custom resnet18, outputs embeddings for circleloss

    def __init__(self, feature_dim= 512, dropout_p=0.5):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT
        self.backbone = resnet34(weights=weights)

        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.feature = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)

        return x

class ResNet50Embedder(nn.Module):# custom resnet18, outputs embeddings for circleloss
    def __init__(self, feature_dim=512, dropout_p= 0.5):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(2048, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.feature = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)

        return x


