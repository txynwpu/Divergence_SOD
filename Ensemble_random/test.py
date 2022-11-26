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
    uncertainty = -1 * torch.sigmoid(pred) * torch.log(torch.sigmoid(pred) + 1e-8)
    return uncertainty

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension')

opt = parser.parse_args()

weight_path = 'models/'  # checkpoint path
save_path = weight_path + 'ens_multi_preds/'

test_data_root = ''        # test data set path

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load(weight_path + 'Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['COME-E', 'COME-H', 'DUTS_Test', 'ECSSD', 'DUT', 'HKU-IS']
time_list = []

for dataset in test_datasets:
    save_path_deter = save_path + 'mj_pred/' + dataset + '/'
    if not os.path.exists(save_path_deter):
        os.makedirs(save_path_deter)

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

    save_path_mean = save_path + 'mean_pred/' + dataset + '/'
    if not os.path.exists(save_path_mean):
        os.makedirs(save_path_mean)

    image_root = test_data_root + 'img/' + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            pred_Mj, pred1,pred2,pred3,pred4,pred5 = generator(image)

        ## uncertainty
        alea_uncer1 = compute_uncertainty(pred1)
        alea_uncer2 = compute_uncertainty(pred2)
        alea_uncer3 = compute_uncertainty(pred3)
        alea_uncer4 = compute_uncertainty(pred4)
        alea_uncer5 = compute_uncertainty(pred5)

        alea_uncertainty = (alea_uncer1+alea_uncer2+alea_uncer3+alea_uncer4+alea_uncer5)/5

        mean_pred = (torch.sigmoid(pred1) + torch.sigmoid(pred2) + torch.sigmoid(pred3) + torch.sigmoid(
            pred4) + torch.sigmoid(pred5)) / 5

        predictive_uncertainty = -1 * mean_pred * torch.log(mean_pred + 1e-8)

        epistemic_uncertainty = predictive_uncertainty - alea_uncertainty

        ## save uncertainty maps
        cv2.imwrite(save_path_aleatoric + name, post_process_uncertainty(alea_uncertainty,HH,WW))
        cv2.imwrite(save_path_epistemic + name, post_process_uncertainty(epistemic_uncertainty, HH, WW))
        cv2.imwrite(save_path_predictive + name, post_process_uncertainty(predictive_uncertainty, HH, WW))



        pred_cat = torch.cat((torch.sigmoid(pred1), torch.sigmoid(pred2), torch.sigmoid(pred3),
                              torch.sigmoid(pred4), torch.sigmoid(pred5)), dim=1)
        pred_mean = torch.mean(pred_cat, 1, keepdim=True)

        pred_mean = post_process_pred_mean(pred_mean, HH, WW)

        cv2.imwrite(save_path_mean + name, pred_mean)


        pred_Mj = post_process_pred(pred_Mj, HH, WW)
        pred1 = post_process_pred(pred1, HH, WW)
        pred2 = post_process_pred(pred2, HH, WW)
        pred3 = post_process_pred(pred3, HH, WW)
        pred4 = post_process_pred(pred4, HH, WW)
        pred5 = post_process_pred(pred5, HH, WW)

        cv2.imwrite(save_path_deter + name, pred_Mj)
        cv2.imwrite(save_path_indiv1 + name, pred1)
        cv2.imwrite(save_path_indiv2 + name, pred2)
        cv2.imwrite(save_path_indiv3 + name, pred3)
        cv2.imwrite(save_path_indiv4 + name, pred4)
        cv2.imwrite(save_path_indiv5 + name, pred5)

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)


