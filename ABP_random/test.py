import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Pred_endecoder, Encode_x
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension')
parser.add_argument('--iter_num', type=int, default=5, help='latent dimension')

opt = parser.parse_args()

weight_path = 'models/'  # checkpoint path
save_path = weight_path + 'abp_multi_preds/'

dataset_path = ''        # test data set path

generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load(weight_path + 'Model_50_gen.pth'))

generator.cuda()
generator.eval()

prior_model = Encode_x(input_channels=3, channels=opt.feat_channel, latent_size=opt.latent_dim)
prior_model.load_state_dict(torch.load(weight_path + 'Model_50_prior.pth'))
prior_model.cuda()
prior_model.eval()

test_datasets = ['COME-E', 'COME-H', 'DUTS_Test', 'ECSSD', 'DUT', 'HKU-IS']
time_list = []

def compute_entropy(pred):
    entropy_loss_fore = -pred * torch.log(pred + 1e-8)
    entropy_loss_back = -(1-pred) * torch.log(1-pred + 1e-8)
    entropy_loss_all = entropy_loss_fore+entropy_loss_back
    return entropy_loss_all

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)

    return eps.mul(std).add_(mu)

for dataset in test_datasets:

    save_path_deter = save_path + 'mj_pred/' + dataset + '/'
    if not os.path.exists(save_path_deter):
        os.makedirs(save_path_deter)

    save_path_mean = save_path + 'pred_mean/' + dataset + '/'
    if not os.path.exists(save_path_mean):
        os.makedirs(save_path_mean)

    save_path_aleatoric = save_path + 'results_aleatoric/' + dataset + '/'
    if not os.path.exists(save_path_aleatoric):
        os.makedirs(save_path_aleatoric)

    save_path_epistemic = save_path + 'results_epistemic/' + dataset + '/'
    if not os.path.exists(save_path_epistemic):
        os.makedirs(save_path_epistemic)

    save_path_predictive = save_path + 'results_predictive/' + dataset + '/'
    if not os.path.exists(save_path_predictive):
        os.makedirs(save_path_predictive)
        
    save_path_indiv1 = save_path + 'pred1/' + dataset + '/'
    if not os.path.exists(save_path_indiv1):
        os.makedirs(save_path_indiv1)

    save_path_indiv2 = save_path + 'pred2/' + dataset + '/'
    if not os.path.exists(save_path_indiv2):
        os.makedirs(save_path_indiv2)

    save_path_indiv3 = save_path + 'pred3/' + dataset + '/'
    if not os.path.exists(save_path_indiv3):
        os.makedirs(save_path_indiv3)

    save_path_indiv4 = save_path + 'pred4/' + dataset + '/'
    if not os.path.exists(save_path_indiv4):
        os.makedirs(save_path_indiv4)

    save_path_indiv5 = save_path + 'pred5/' + dataset + '/'
    if not os.path.exists(save_path_indiv5):
        os.makedirs(save_path_indiv5)

    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        mean_pred = 0
        alea_uncertainty = 0
        start = time.time()
        mu, logsig = prior_model(image)

        deter_pred = generator.forward(image)
        res = F.upsample(deter_pred, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_deter + name, res)

        for iter in range(opt.iter_num):
            # print(iter)

            z_noise = reparametrize(mu, logsig)

            # z_noise = torch.randn(image.shape[0], opt.latent_dim).cuda()
            torch.cuda.synchronize()

            generator_pred = generator.forward(image, z_noise)
            res = F.upsample(generator_pred, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + 'pred' + str(iter+1) + '/' + dataset + '/' + name, res)
            
            mean_pred = mean_pred + torch.sigmoid(generator_pred)
            preds = torch.sigmoid(generator_pred)
            cur_alea = -1 * preds * torch.log(preds + 1e-8)
            # cur_alea = compute_entropy(preds)
            alea_uncertainty = alea_uncertainty + cur_alea

        mean_prediction = mean_pred/opt.iter_num
        alea_uncertainty = alea_uncertainty/opt.iter_num
        predictive_uncertainty = -1 * mean_prediction * torch.log(mean_prediction + 1e-8)
        # predictive_uncertainty = compute_entropy(mean_prediction)
        epistemic_uncertainty = predictive_uncertainty-alea_uncertainty

        res = F.upsample(mean_prediction, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_mean + name, res)

        res = F.upsample(alea_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_aleatoric + name, res)

        res = F.upsample(epistemic_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_epistemic + name, res)

        res = F.upsample(predictive_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_predictive + name, res)

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
