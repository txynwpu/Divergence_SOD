import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Pred_endecoder, Encode_x
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_pri', type=float, default=1e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension')
parser.add_argument('--langevin_step_num_gen', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--sigma_gen', type=float, default=0.3, help='ebm initial sigma')
parser.add_argument('--langevin_s', type=float, default=0.1, help='ebm initial sigma')
opt = parser.parse_args()


print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

prior_model = Encode_x(input_channels=3, channels=opt.feat_channel, latent_size=opt.latent_dim)
prior_model.cuda()
prior_model_params = prior_model.parameters()
prior_model_optimizer = torch.optim.Adam(prior_model_params, opt.lr_pri)

image_root = ''       # DUTS Train image path
gt_root = ''          # DUTS Train majority voting gt path
gt_root1 = ''         # DUTS Train diverse gt1 path
gt_root2 = ''         # DUTS Train diverse gt2 path
gt_root3 = ''         # DUTS Train diverse gt3 path
gt_root4 = ''         # DUTS Train diverse gt4 path
gt_root5 = ''         # DUTS Train diverse gt5 path

train_loader = get_loader(image_root, gt_root, gt_root1, gt_root2, gt_root3, gt_root4, gt_root5, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduction='sum')
size_rates = [1]  # multi-scale training


def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss_focal_loss(pred, mask, weight):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (((1-weight)**opt.focal_lamda)*weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred_init(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)

def no_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()

def yes_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def KL_computation(inf_mean, inf_logvar):
    KLD = -1 * 0.5 * torch.sum(1 + inf_logvar - inf_mean.pow(2) - inf_logvar.exp())
    return KLD

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)

    return eps.mul(std).add_(mu)

def visualize_all_pred(mj_pred, pred1,pred2,pred3,pred4,pred5):
    for kk in range(mj_pred.shape[0]):
        pred_mj, pred1, pred2, pred3, pred4, pred5 = mj_pred[kk, :, :, :], pred1[kk, :, :, :], pred2[kk, :, :, :], pred3[kk, :, :, :], pred4[kk, :, :, :], pred5[kk, :, :, :]
        pred_mj = (pred_mj.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        pred1 = (pred1.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        pred2 = (pred2.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred3 = (pred3.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred4 = (pred4.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred5 = (pred5.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_mj, pred1, pred2, pred3, pred4, pred5])
        save_path = './temp/'
        name = '{:02d}_preds.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

def visualize_all_pred2(mj_pred, pred1):
    for kk in range(mj_pred.shape[0]):
        pred_mj, pred_sto = mj_pred[kk, :, :, :], pred1[kk, :, :, :]
        pred_mj = (pred_mj.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        pred_sto = (pred_sto.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_mj, pred_sto])
        save_path = './temp/'
        name = '{:02d}_preds2.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

def visualize_all_gt2(mj_pred, pred1):
    for kk in range(mj_pred.shape[0]):
        pred_mj, pred_sto = mj_pred[kk, :, :, :], pred1[kk, :, :, :]
        pred_mj = (pred_mj.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred_sto = (pred_sto.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_mj, pred_sto])
        save_path = './temp/'
        name = '{:02d}_gts2.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

def visualize_all_pred_gt2(mj_pred, pred1, mj_gts, gts):
    for kk in range(mj_pred.shape[0]):
        pred_mj, pred_sto, gt_mj, gt = mj_pred[kk, :, :, :], pred1[kk, :, :, :], mj_gts[kk, :, :, :], gts[kk, :, :, :]
        pred_mj = (pred_mj.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred_sto = (pred_sto.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        gt_mj = (gt_mj.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        gt = (gt.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_mj, pred_sto, gt_mj, gt])
        save_path = './temp/'
        name = '{:02d}_pred_gts2.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

def visualize_all_gt(mj_pred, pred1, pred2, pred3, pred4, pred5):
    for kk in range(mj_pred.shape[0]):
        pred_mj, pred1, pred2, pred3, pred4, pred5 = mj_pred[kk, :, :, :], pred1[kk, :, :, :], pred2[kk, :, :,
                                                                                               :], pred3[kk, :, :,
                                                                                                   :], pred4[kk, :, :,
                                                                                                       :], pred5[kk, :,
                                                                                                           :, :]
        pred_mj = (pred_mj.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred1 = (pred1.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred2 = (pred2.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred3 = (pred3.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred4 = (pred4.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred5 = (pred5.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred_mj, pred1, pred2, pred3, pred4, pred5])
        save_path = './temp/'
        name = '{:02d}_gts.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

print("Let's go!")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    prior_model.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            prior_model_optimizer.zero_grad()
            
            images, gts, gts1, gts2, gts3, gts4, gts5 = pack
            images = Variable(images)
            gts = Variable(gts)
            gts1 = Variable(gts1)
            gts2 = Variable(gts2)
            gts3 = Variable(gts3)
            gts4 = Variable(gts4)
            gts5 = Variable(gts5)
            images = images.cuda()
            gts = gts.cuda()
            gts1 = gts1.cuda()
            gts2 = gts2.cuda()
            gts3 = gts3.cuda()
            gts4 = gts4.cuda()
            gts5 = gts5.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts1 = F.upsample(gts1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts2 = F.upsample(gts2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts3 = F.upsample(gts3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts4 = F.upsample(gts4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts5 = F.upsample(gts5, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            mu,logsig = prior_model(images)
            kld = KL_computation(mu, logsig)
            z_prior = reparametrize(mu, logsig)
            kld.backward()
            prior_model_optimizer.step()

            gt_cur = gts

            # z_noise = train_z[index_batch]
            z_noise = Variable(z_prior, requires_grad=True)
            z_noise_preds = [z_noise.clone() for _ in range(opt.langevin_step_num_gen + 1)]
            ## update latent variable z of generator with MCMC
            for kk in range(opt.langevin_step_num_gen):
                z_noise = Variable(z_noise_preds[kk], requires_grad=True)
                z_noise = z_noise.cuda()
                noise = torch.randn(z_noise.size())
                noise = noise.cuda()
                gen_res = generator(images, z_noise)
                random_rot_index = torch.randint(0, 5, (1,))
                if random_rot_index == 0:
                    gt_cur = gts1
                elif random_rot_index == 1:
                    gt_cur = gts2
                elif random_rot_index == 2:
                    gt_cur = gts3
                elif random_rot_index == 3:
                    gt_cur = gts4
                else:
                    gt_cur = gts5

                gen_loss = 1 / (2.0 * opt.sigma_gen * opt.sigma_gen) * mse_loss(
                    torch.sigmoid(gen_res), gt_cur)
                gen_loss.backward(torch.ones(gen_loss.size()).cuda())
                grad = z_noise.grad
                z_noise = z_noise + 0.5 * opt.langevin_s * opt.langevin_s * grad
                z_noise += opt.langevin_s * noise
                z_noise_preds[kk + 1] = z_noise
            z_noise_post = z_noise_preds[-1]


            pred_deter = generator(images)
            pred_sto = generator(images,z_noise_post)

            random_rot_index = torch.randint(0, 2, (1,))
            if random_rot_index == 1:
                print('deterministic updating')
                loss_all = structure_loss(pred_deter, gts)
            else:
                print('stochastic updating')
                loss_all = structure_loss(pred_sto, gt_cur)


            loss_all.backward()
            generator_optimizer.step()

            visualize_all_pred_gt2(torch.sigmoid(pred_deter), torch.sigmoid(pred_sto), gts, gt_cur)

            if rate == 1:
                loss_record.update(loss_all.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(prior_model_optimizer, opt.lr_pri, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(prior_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_prior.pth')
