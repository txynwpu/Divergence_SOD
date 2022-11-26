import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from torchvision import transforms
from model.ResNet_models import Pred_endecoder

from data import test_dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def post_process_pred_mean(pred1, height, width):
    pred1 = F.upsample(pred1, size=[width, height], mode='bilinear', align_corners=False)
    pred1 = pred1.data.cpu().numpy().squeeze()
    pred1 = 255 * (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)
    return pred1

def post_process_pred(pred1, height, width):
    pred1 = F.upsample(pred1, size=[width, height], mode='bilinear', align_corners=False)
    # pred1 = torch.sigmoid(pred1).squeeze()
    pred1 = pred1.sigmoid().data.cpu().numpy().squeeze()
    pred1 = 255 * (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)

    return pred1

def post_process_uncertainty(pred1, height, width):
    pred1 = F.upsample(pred1, size=[width, height], mode='bilinear', align_corners=False)
    pred1 = pred1.data.cpu().numpy().squeeze()
    pred1 = 255 * (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)
    pred1 = pred1.astype(np.uint8)
    pred1 = cv2.applyColorMap(pred1, cv2.COLORMAP_JET)
    return pred1

def compute_uncertainty(pred):
    uncertainty = -1 * pred * torch.log(pred + 1e-8)
    return uncertainty

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension')
parser.add_argument('--forward_iter', type=int, default=5, help='number of iterations of gnerator forward')
opt = parser.parse_args()

weight_path =  'models/'  # checkpoint path
save_path = weight_path + 'vae_multi_preds/'

test_data_root = ''        # test data set path

generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load(weight_path + 'Model_50_gen.pth'))

generator.cuda()
generator.eval()
test_datasets = ['COME-E', 'COME-H', 'DUTS_Test', 'ECSSD', 'DUT', 'HKU-IS']

time_list = []

for dataset in test_datasets:
    save_path_deter = save_path + 'results/' + dataset + '/'
    if not os.path.exists(save_path_deter):
        os.makedirs(save_path_deter)

    save_path_mean = save_path + 'results_mean/' + dataset + '/'
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


    image_root = test_data_root + 'img/' + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size), desc=dataset):
        preds = []
        alea = []
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        torch.cuda.synchronize()
        start = time.time()

        mj_pred, pred_prior1 = generator(image, training=False)
        pred_Mj = post_process_pred(mj_pred, HH, WW)
        cv2.imwrite(save_path_deter + name, pred_Mj)

        for uu in range(opt.forward_iter):
            sto_pred = torch.sigmoid(generator.forward(image, training=False)[1])
            alea.append(compute_uncertainty(sto_pred))
            preds.append(sto_pred)

            res = post_process_pred(sto_pred, HH, WW)
            cv2.imwrite(save_path + 'pred' + str(uu + 1) + '/' + dataset + '/' + name, res)


        preds = torch.cat(preds, dim=1)
        pred_mean = torch.mean(preds, 1, keepdim=True)
        predictive_uncertainty = compute_uncertainty(pred_mean)

        alea = torch.cat(alea, dim=1)
        aleatoric_uncertainty = torch.mean(alea, 1, keepdim=True)

        epistemic_uncertainty = predictive_uncertainty-aleatoric_uncertainty


        res = post_process_pred_mean(pred_mean, HH, WW)
        cv2.imwrite(save_path_mean + name, res)

        ## save uncertainty maps
        cv2.imwrite(save_path_aleatoric + name, post_process_uncertainty(aleatoric_uncertainty, HH, WW))
        cv2.imwrite(save_path_epistemic + name, post_process_uncertainty(epistemic_uncertainty, HH, WW))
        cv2.imwrite(save_path_predictive + name, post_process_uncertainty(predictive_uncertainty, HH, WW))

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)


