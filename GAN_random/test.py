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


def post_process_pred(pred1, height, width):
    pred1 = F.upsample(pred1, size=[width, height], mode='bilinear', align_corners=False)
    # pred1 = torch.sigmoid(pred1).squeeze()
    pred1 = pred1.sigmoid().data.cpu().numpy().squeeze()
    pred1 = 255 * (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)

    return pred1
    
def compute_uncertainty(pred):
    uncertainty = -1 * torch.sigmoid(pred) * torch.log(torch.sigmoid(pred) + 1e-8)
    return uncertainty

def post_process_uncertainty(pred1, height, width):
    pred1 = F.upsample(pred1, size=[width, height], mode='bilinear', align_corners=False)
    pred1 = pred1.data.cpu().numpy().squeeze()
    pred1 = 255 * (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)
    pred1 = pred1.astype(np.uint8)
    pred1 = cv2.applyColorMap(pred1, cv2.COLORMAP_JET)
    return pred1


weight_path = 'models/'  # checkpoint path
save_path = weight_path + 'gan_multi_preds/'
# writer = SummaryWriter(save_path + '/log')


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension')

opt = parser.parse_args()


test_data_root = ''        # test data set path

generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load(weight_path + 'Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['COME-E', 'COME-H', 'DUTS_Test', 'ECSSD', 'DUT', 'HKU-IS']

time_list = []

for dataset in test_datasets:
    print(dataset)
    save_path1 = save_path + 'mj_pred/' + dataset + '/'
    save_path2 = save_path + 'pred1/' + dataset + '/'
    save_path3 = save_path + 'pred2/' + dataset + '/'
    save_path4 = save_path + 'pred3/' + dataset + '/'
    save_path5 = save_path + 'pred4/' + dataset + '/'
    save_path6 = save_path + 'pred5/' + dataset + '/'
    save_path7 = save_path + 'mean_pred/' + dataset + '/'

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    if not os.path.exists(save_path3):
        os.makedirs(save_path3)
    if not os.path.exists(save_path4):
        os.makedirs(save_path4)
    if not os.path.exists(save_path5):
        os.makedirs(save_path5)
    if not os.path.exists(save_path6):
        os.makedirs(save_path6)
    if not os.path.exists(save_path7):
        os.makedirs(save_path7)
        
    save_path_aleatoric = save_path + 'results_aleatoric/' + dataset + '/'
    if not os.path.exists(save_path_aleatoric):
        os.makedirs(save_path_aleatoric)

    save_path_epistemic = save_path + 'results_epistemic/' + dataset + '/'
    if not os.path.exists(save_path_epistemic):
        os.makedirs(save_path_epistemic)

    save_path_predictive = save_path + 'results_predictive/' + dataset + '/'
    if not os.path.exists(save_path_predictive):
        os.makedirs(save_path_predictive)


    image_root = test_data_root + 'img/' + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            mj_pred, pred1, pred2, pred3, pred4, pred5, latent_loss = generator(image)
              # Inference and get the last one of the output list
        
        mj_pred = post_process_pred(mj_pred, HH, WW)
        cv2.imwrite(save_path1 + name, mj_pred)

        
        pred_cat = torch.cat((torch.sigmoid(pred1), torch.sigmoid(pred2), torch.sigmoid(pred3),
                              torch.sigmoid(pred4), torch.sigmoid(pred5)), dim=1)
        pred_mean = torch.mean(pred_cat, 1, keepdim=True)

        pred_mean = F.upsample(pred_mean, size=[WW, HH], mode='bilinear', align_corners=False)

          # generate mean prediction
        pred_mean = pred_mean.data.cpu().numpy().squeeze()
        pred_mean = 255 * (pred_mean - pred_mean.min()) / (pred_mean.max() - pred_mean.min() + 1e-8)

        cv2.imwrite(save_path7 + name, pred_mean)

        ## uncertainty
        alea_uncer1 = compute_uncertainty(pred1)
        alea_uncer2 = compute_uncertainty(pred2)
        alea_uncer3 = compute_uncertainty(pred3)
        alea_uncer4 = compute_uncertainty(pred4)
        alea_uncer5 = compute_uncertainty(pred5)

        alea_uncertainty = (alea_uncer1 + alea_uncer2 + alea_uncer3 + alea_uncer4 + alea_uncer5) / 5

        mean_pred = (torch.sigmoid(pred1) + torch.sigmoid(pred2) + torch.sigmoid(pred3) + torch.sigmoid(
            pred4) + torch.sigmoid(pred5)) / 5

        predictive_uncertainty = -1 * mean_pred * torch.log(mean_pred + 1e-8)

        epistemic_uncertainty = predictive_uncertainty - alea_uncertainty

        ## save uncertainty maps
        cv2.imwrite(save_path_aleatoric + name, post_process_uncertainty(alea_uncertainty, HH, WW))
        cv2.imwrite(save_path_epistemic + name, post_process_uncertainty(epistemic_uncertainty, HH, WW))
        cv2.imwrite(save_path_predictive + name, post_process_uncertainty(predictive_uncertainty, HH, WW))

        ## save diverse preds
        pred1 = post_process_pred(pred1, HH, WW)
        pred2 = post_process_pred(pred2, HH, WW)
        pred3 = post_process_pred(pred3, HH, WW)
        pred4 = post_process_pred(pred4, HH, WW)
        pred5 = post_process_pred(pred5, HH, WW)

        cv2.imwrite(save_path2 + name, pred1)
        cv2.imwrite(save_path3 + name, pred2)
        cv2.imwrite(save_path4 + name, pred3)
        cv2.imwrite(save_path5 + name, pred4)
        cv2.imwrite(save_path6 + name, pred5)


        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)

