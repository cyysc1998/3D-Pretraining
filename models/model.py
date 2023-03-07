import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
sys.path.append('./models')
from dgcnn import DGCNN
from tools import MLP


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used 
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# DenseNet Decoder

class _DenseLayer(nn.Sequential):
    def __init__(self, args, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        if args.extra == 'adain':
            self.add_module("norm1", AdaptiveInstanceNorm2d(num_input_features))
        elif args.extra == 'concat':
            self.add_module("norm1", nn.BatchNorm1d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        if args.extra == 'adain':
            self.add_module("norm2", AdaptiveInstanceNorm2d(bn_size * growth_rate))
        elif args.extra == 'concat':
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=1, stride=1, bias=False))
    
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = super(_DenseLayer, self).forward(x)
        x = torch.cat([x, x1], dim=1)

        return x

class DenseNet(nn.Module):
    def __init__(self, args, num_layers, num_input_features, bn_size, growth_rate):
        super(DenseNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(num_layers):
            layer = _DenseLayer(args, num_input_features + i * growth_rate, growth_rate, bn_size)
            self.model.add_module("Denselayer%d" % (i), layer)
    def forward(self, x):
        x = self.model(x)
        return x

# InsResDecoder

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)


class InsResBlock(nn.Module):
    def __init__(self, C_in, dropout):
        super(InsResBlock, self).__init__()
        model = []
        model += [nn.Conv1d(C_in, 512, kernel_size=1)]
        model += [nn.InstanceNorm1d(512)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv1d(512, C_in, kernel_size=1)]
        model += [nn.InstanceNorm1d(C_in)]
        if dropout:
            model += [nn.Dropout(p=0.5)]
        
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class InsResDecoder(nn.Module):
    def __init__(self, args, C_in, C_out):
        super(InsResDecoder, self).__init__()

        self.decoder_list = [256, 128, 64]
        self.y_len = 64

        self.InsResBlock1 = InsResBlock(self.decoder_list[0] + self.y_len, args.dropout)
        self.InsResBlock2 = InsResBlock(self.decoder_list[1] + self.y_len, args.dropout)
        self.InsResBlock3 = InsResBlock(self.decoder_list[2] + self.y_len, args.dropout)

        self.ConvBlock1 = nn.Conv1d(self.decoder_list[0] + self.y_len, self.decoder_list[0] // 2, kernel_size=1)
        self.ConvBlock2 = nn.Conv1d(self.decoder_list[1] + self.y_len, self.decoder_list[1] // 2, kernel_size=1)
        self.ConvBlock3 = nn.Conv1d(self.decoder_list[2] + self.y_len, C_out, kernel_size=1)

        self.y_conv = nn.Conv1d(args.extra_len, self.y_len, kernel_size=1)

    def forward(self, x, y):
        y = self.y_conv(y) # (B, extra_len, N) -> (B, 64, N)
        
        x = self.InsResBlock1(torch.cat([x, y], dim=1))
        x = self.ConvBlock1(x)
        x = self.InsResBlock2(torch.cat([x, y], dim=1))
        x = self.ConvBlock2(x)
        x = self.InsResBlock3(torch.cat([x, y], dim=1))
        x = self.ConvBlock3(x)
        
        return x


class SpatialWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=1,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.relu(self.conv1(out))
        out = self.sigmoid(self.conv2(out))
        return x * out


# Model

class ConsNet(nn.Module):
    def __init__(self, args, seg_num_all):
        super(ConsNet, self).__init__()
        self.args = args
        self.dgcnn = DGCNN(args, seg_num_all)
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

        if args.decoder == 'densegcn':
            self.decoder = DenseNet(args, args.nlayers, 256, 4, 128)
            self.extra = args.extra

            if args.extra == 'adain':
                self.output_dim = []
                for i in range(args.nlayers):
                    self.output_dim.append(256 + 128 * i)
                    self.output_dim.append(512)
                self.mlp_w = MLP(args.extra_len*args.num_points, 64 * args.nlayers * args.nlayers + 704 * args.nlayers, 256, 3)
                self.mlp_b = MLP(args.extra_len*args.num_points, 64 * args.nlayers * args.nlayers + 704 * args.nlayers, 256, 3)
                
            elif args.extra == 'concat':
                self.y_encoder = nn.Conv1d(args.extra_len, 64, kernel_size=1)
                self.y_convert = nn.Conv1d(64, 64, kernel_size=1)
                self.x_convert = nn.Conv1d(320, 256, kernel_size=1)
                self.dp1 = nn.Dropout(p=0.5)
                self.dp2 = nn.Dropout(p=0.5)
            else:
                raise NotImplementedError

            self.conv = nn.Conv1d(256 + 128 * args.nlayers, 3, kernel_size=1)

        elif args.decoder == 'insresblock':
            self.decoder = InsResDecoder(args, 256, 3)

        else:
            raise NotImplementedError

        self.cg = SpatialWeighting(3)


    def forward(self, x, y, l):
        x, embed = self.dgcnn(x, l) # (B, 256, N)

        if self.args.decoder == 'densegcn':
            if self.extra == 'adain':
                # y shape(B, N, extra_len)
                y = y.permute(0, 2, 1).reshape(y.size(0), -1) 
            
                adain_params_w = self.mlp_w(y)
                adain_params_b = self.mlp_b(y)
                self.assign_adain_params(adain_params_w, adain_params_b, self.decoder)
            elif self.extra == 'concat':
                y = self.y_encoder(y.permute(0, 2, 1)) # (B, 64, N)
                y = self.dp1(y) 
                y = self.y_convert(y)
                y = self.dp2(y)
                x = torch.cat([x, y], dim=1) # (B, 320, N)
                x = self.x_convert(x) # (B, 320, N) -> (B, 256, N)
            else:
                raise NotImplementedError

            x = self.decoder(x)
            x = self.conv(x)

        elif self.args.decoder == 'insresblock':
            y = y.permute(0, 2, 1) # (B, N, extra_len) -> (B, extra_len, N)
            x = self.decoder(x, y)

        else:
            raise NotImplementedError

        x = self.cg(x)

        return x, embed, self.logit_scale
    
    
    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:,:dim[0]].contiguous()
                std = adain_params_w[:,:dim[0]].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1)>dim[0] :  #Pop the parameters
                    adain_params_b = adain_params_b[:,dim[0]:]
                    adain_params_w = adain_params_w[:,dim[0]:]
                    dim = dim[1:]
    
