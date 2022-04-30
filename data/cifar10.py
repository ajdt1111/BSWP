from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform_train_shuffle = transforms.Compose([
            #AddGaussianNoise(mean=0, variance=0.03, amplitude=1),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #img_shuffle(block_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])

        transform_test_shuffle = transforms.Compose([
            img_shuffle(block_size=16),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset_shuffle = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train_shuffle)


        self.loader_train_shuffle = DataLoader(
            trainset_shuffle, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=pin_memory
            )


        testset_shuffle = CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test_shuffle)
        self.loader_test_shuffle = DataLoader(
            testset_shuffle, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class img_shuffle(object):

    def __init__(self, block_size=3):
        self.block_size = block_size

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        x1 = self.block_size * int(h / self.block_size)
        y1 = self.block_size * int(w / self.block_size)

        # shuffle
        order_list = list(range(self.block_size ** 2))  # [0,1,2,3]
        np.random.shuffle(order_list)
        #data = np.array(img.resize([x1, y1]))
        data=img
        data_new = np.ones(data.shape)
        for idx, item1 in enumerate(order_list):
            ind_i = int(item1 // self.block_size)
            ind_j = int(item1 % self.block_size)
            index_i = int(idx // self.block_size)
            index_j = int(idx % self.block_size)
            x_length = int(x1 / self.block_size)
            y_length = int(y1 / self.block_size)
            tmp = data[y_length * ind_j:y_length * (ind_j + 1), x_length * ind_i:x_length * (ind_i + 1), :]
            data_new[y_length * index_j:y_length * (index_j + 1), x_length * index_i:x_length * (index_i + 1), :] = tmp
        #x = Image.fromarray(np.uint8(data_new))
        img = Image.fromarray(data_new.astype('uint8')).convert('RGB')
        return img

