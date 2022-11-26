import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label, label1, label2, label3, label4, label5):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        label1 = label1.transpose(Image.FLIP_LEFT_RIGHT)
        label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)
        label3 = label3.transpose(Image.FLIP_LEFT_RIGHT)
        label4 = label4.transpose(Image.FLIP_LEFT_RIGHT)
        label5 = label5.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, label1, label2, label3, label4, label5


def randomCrop(img, label, label1, label2, label3, label4, label5):
    border = 30
    image_width = img.size[0]
    image_height = img.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return img.crop(random_region), label.crop(random_region), label1.crop(random_region), label2.crop(
        random_region), label3.crop(random_region), label4.crop(random_region), label5.crop(random_region)

def randomRotation(img, label, label1, label2, label3, label4, label5):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        label1 = label1.rotate(random_angle, mode)
        label2 = label2.rotate(random_angle, mode)
        label3 = label3.rotate(random_angle, mode)
        label4 = label4.rotate(random_angle, mode)
        label5 = label5.rotate(random_angle, mode)

    return img, label, label1, label2, label3, label4, label5


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0, sigma=0.15):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomGaussian1(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, gt_root1, gt_root2, gt_root3, gt_root4, gt_root5, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts1 = [gt_root1 + f for f in os.listdir(gt_root1) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts2 = [gt_root2 + f for f in os.listdir(gt_root2) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts3 = [gt_root3 + f for f in os.listdir(gt_root3) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts4 = [gt_root4 + f for f in os.listdir(gt_root4) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts5 = [gt_root5 + f for f in os.listdir(gt_root5) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gts1 = sorted(self.gts1)
        self.gts2 = sorted(self.gts2)
        self.gts3 = sorted(self.gts3)
        self.gts4 = sorted(self.gts4)
        self.gts5 = sorted(self.gts5)


        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt1 = self.binary_loader(self.gts1[index])
        gt2 = self.binary_loader(self.gts2[index])
        gt3 = self.binary_loader(self.gts3[index])
        gt4 = self.binary_loader(self.gts4[index])
        gt5 = self.binary_loader(self.gts5[index])
        image, gt, gt1, gt2, gt3, gt4, gt5 = cv_random_flip(image, gt, gt1, gt2, gt3, gt4, gt5)
        image, gt, gt1, gt2, gt3, gt4, gt5 = randomCrop(image, gt, gt1, gt2, gt3, gt4, gt5)
        image, gt, gt1, gt2, gt3, gt4, gt5  = randomRotation(image, gt, gt1, gt2, gt3, gt4, gt5)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        gt1 = randomPeper(gt1)
        gt2 = randomPeper(gt2)
        gt3 = randomPeper(gt3)
        gt4 = randomPeper(gt4)
        gt5 = randomPeper(gt5)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gt1 = self.gt_transform(gt1)
        gt2 = self.gt_transform(gt2)
        gt3 = self.gt_transform(gt3)
        gt4 = self.gt_transform(gt4)
        gt5 = self.gt_transform(gt5)
        return image, gt, gt1, gt2, gt3, gt4, gt5


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.gts1)
        assert len(self.images) == len(self.gts2)
        assert len(self.images) == len(self.gts3)
        assert len(self.images) == len(self.gts4)
        assert len(self.images) == len(self.gts5)
        images = []
        gts = []
        gts1 = []
        gts2 = []
        gts3 = []
        gts4 = []
        gts5 = []
        for img_path, gt_path, gt_path1, gt_path2, gt_path3, gt_path4, gt_path5 in zip(self.images, self.gts, self.gts1, self.gts2, self.gts3, self.gts4, self.gts5):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            gt1 = Image.open(gt_path1)
            gt2 = Image.open(gt_path2)
            gt3 = Image.open(gt_path3)
            gt4 = Image.open(gt_path4)
            gt5 = Image.open(gt_path5)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                gts1.append(gt_path1)
                gts2.append(gt_path2)
                gts3.append(gt_path3)
                gts4.append(gt_path4)
                gts5.append(gt_path5)

        self.images = images
        self.gts = gts
        self.gts1 = gts1
        self.gts2 = gts2
        self.gts3 = gts3
        self.gts4 = gts4
        self.gts5 = gts5

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, gt_root1, gt_root2, gt_root3, gt_root4, gt_root5, batchsize, trainsize, shuffle=True, num_workers=2, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, gt_root1, gt_root2, gt_root3, gt_root4, gt_root5, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# test dataset and loader

class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)  if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])

        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


