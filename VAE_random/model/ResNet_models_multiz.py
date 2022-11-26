import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np


class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.tanh = torch.nn.Tanh()


        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        # output = output.view(-1, self.channel * 8 * 11 * 11)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # mu = self.fc1(output)
        # logvar = self.fc2(output)
        # dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # return dist, mu, logvar

class Encoder_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.tanh = torch.nn.Tanh()

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        # print(output.size())
        if x.shape[2] == 256:
            # print('************************256********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)


            return dist, mu, logvar
        elif x.shape[2] == 352:
            # print('************************352********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)


            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            # print(mu)
            # print(logvar)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)


            return dist, mu, logvar

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class Pred_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        super(Pred_endecoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.x_encoder = Encoder_x(3, channel, latent_dim)
        self.xy_encoder = Encoder_x(4, channel, latent_dim)
        self.majority_dec =  Pred_decoder_majority(channel)
        self.prior_dec = Pred_decoder(channel,latent_dim)
        self.post_dec = Pred_decoder(channel, latent_dim)

        self.fc_mean = nn.Linear(latent_dim*5, latent_dim)
        self.fc_logsigma = nn.Linear(latent_dim * 5, latent_dim)

        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, y=None, y1=None, y2=None, y3=None, y4=None, y5=None):
        raw_x = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        mj_pred = self.majority_dec(x1,x2,x3,x4)

        if y != None:
            _, muxy1, logvarxy1 = self.xy_encoder(torch.cat((raw_x, y1), 1))
            _, muxy2, logvarxy2 = self.xy_encoder(torch.cat((raw_x, y2), 1))
            _, muxy3, logvarxy3 = self.xy_encoder(torch.cat((raw_x, y3), 1))
            _, muxy4, logvarxy4 = self.xy_encoder(torch.cat((raw_x, y4), 1))
            _, muxy5, logvarxy5 = self.xy_encoder(torch.cat((raw_x, y5), 1))

            muxy_cat = torch.cat((muxy1, muxy2, muxy3, muxy4, muxy5), 1)
            logvarxy_cat = torch.cat((logvarxy1, logvarxy2, logvarxy3, logvarxy4, logvarxy5), 1)
            muxy = self.fc_mean(muxy_cat)
            logvarxy = self.fc_logsigma(logvarxy_cat)
            self.posterior = Independent(Normal(loc=muxy, scale=torch.exp(logvarxy)), 1)

            self.prior, mux, logvarx = self.x_encoder(raw_x)
            latent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            z_noise_prior = self.reparametrize(mux, logvarx)
            pred_prior1 = self.prior_dec(x1, x2, x3, x4, z_noise_prior)
            z_noise_prior = self.reparametrize(mux, logvarx)
            pred_prior2 = self.prior_dec(x1, x2, x3, x4, z_noise_prior)
            z_noise_prior = self.reparametrize(mux, logvarx)
            pred_prior3 = self.prior_dec(x1, x2, x3, x4, z_noise_prior)
            z_noise_prior = self.reparametrize(mux, logvarx)
            pred_prior4 = self.prior_dec(x1, x2, x3, x4, z_noise_prior)
            z_noise_prior = self.reparametrize(mux, logvarx)
            pred_prior5 = self.prior_dec(x1, x2, x3, x4, z_noise_prior)

            z_noise_post = self.reparametrize(muxy, logvarxy)
            pred_post = self.post_dec(x1, x2, x3, x4, z_noise_post)

            return mj_pred, pred_prior1, pred_prior2, pred_prior3, pred_prior4, pred_prior5, pred_post, latent_loss
        else:

            self.prior, mux, logvarx = self.x_encoder(raw_x)
            z_noise_prior = self.reparametrize(mux, logvarx)
            # print(z_noise_prior)
            pred_prior = self.prior_dec(x1, x2, x3, x4, z_noise_prior)
            return mj_pred, pred_prior


    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

class Pred_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        super(Pred_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.spatial_axes = [2, 3]
        self.conv_z = BasicConv2d(channel + latent_dim, channel, kernel_size=3, padding=1)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)


    def forward(self, x1,x2,x3,x4, z):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)


        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, conv4_feat.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, conv4_feat.shape[self.spatial_axes[1]])

        conv4_feat = torch.cat((conv4_feat, z), 1)
        conv4_feat = self.conv_z(conv4_feat)

        z = self.up_2(z)

        conv3_feat = torch.cat((conv3_feat, z), 1)
        conv3_feat = self.conv_z(conv3_feat)

        z = self.up_2(z)

        conv2_feat = torch.cat((conv2_feat, z), 1)
        conv2_feat = self.conv_z(conv2_feat)

        z = self.up_2(z)

        conv1_feat = torch.cat((conv1_feat, z), 1)
        conv1_feat = self.conv_z(conv1_feat)

        conv4_feat = self.path4(conv4_feat)
        conv43 = self.path3(conv4_feat, conv3_feat)
        conv432 = self.path2(conv43, conv2_feat)
        conv4321 = self.path1(conv432, conv1_feat)

        pred = self.output_conv(conv4321)
        return pred


class Pred_decoder_majority(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Pred_decoder_majority, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x1,x2,x3,x4):
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        conv4_feat = self.path4(conv4_feat)
        conv43 = self.path3(conv4_feat, conv3_feat)
        conv432 = self.path2(conv43, conv2_feat)
        conv4321 = self.path1(conv432, conv1_feat)

        pred = self.output_conv(conv4321)
        return pred
