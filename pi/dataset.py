import numpy as np, torch
from torchvision import transforms, datasets
import torchvision
from PIL import Image

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, std=0.15):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        return noisy_tensor

def split_label_unlabel_valid(labels, num_label, num_classes):
    labels_per_class = num_label // num_classes
    labeled_idx = []
    unlabeled_idx = []
    valid_idx = []
    for i in range(num_classes):
        idx = np.where(np.array(labels) == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:labels_per_class])
        valid_idx.extend(idx[labels_per_class:100 + labels_per_class])
        unlabeled_idx.extend(idx[labels_per_class + 100:])
    return labeled_idx, unlabeled_idx, valid_idx


def one_hot(label, num_classes):
    return torch.eye(num_classes)[label]


def get_cifar10(args):
    trainset = torchvision.datasets.CIFAR10(root=args.root,
                                            train=True,
                                            download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.targets, args.num_labels, 10)

    train_labeled_dataset = CIFAR10SSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = CIFAR10SSL(root=args.root,
                                         indexs=unlabeled_idx,
                                         is_labeled=False)

    valid_dataset = CIFAR10SSL(root=args.root, indexs=valid_idx, is_valid=True)

    test_dataset = CIFAR10SSL(root=args.root, train=False)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_cifar100(args):

    trainset = torchvision.datasets.CIFAR100(root=args.root,
                                             train=True,
                                             download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.targets, args.num_labels, 100)

    train_labeled_dataset = CIFAR100SSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = CIFAR100SSL(root=args.root,
                                          indexs=unlabeled_idx,
                                          is_labeled=False)

    valid_dataset = CIFAR100SSL(root=args.root,
                                indexs=valid_idx,
                                is_valid=True)

    test_dataset = CIFAR100SSL(root=args.root, train=False)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_svhn(args):
    trainset = torchvision.datasets.SVHN(root=args.root,
                                         split="train",
                                         download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.labels, args.num_labels, 10)

    train_labeled_dataset = SVHNSSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = SVHNSSL(root=args.root,
                                      indexs=unlabeled_idx,
                                      is_labeled=False)

    valid_dataset = SVHNSSL(root=args.root, indexs=valid_idx, is_valid=True)

    test_dataset = SVHNSSL(root=args.root, split="test")

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset



class tranformSSL:
    def __init__(self, mean, std):
        self.noisy = AddGaussianNoise(mean=0.0, std=0.15)
        self.train_transfrom = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            self.noisy,
            transforms.Normalize(mean=mean, std=std)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

class CIFAR10SSL(datasets.CIFAR10):

    def __init__(self,
                 root,
                 indexs=None,
                 train=True,
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        self.transform = tranformSSL(mean, std)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if not self.train or self.is_valid:
            return self.transform.test_transform(img), one_hot(target, 10)

        trans_img = self.weak(img)

        if self.is_labeled:
            return self.transform.train_transfrom(trans_img), one_hot(target, 10)

        return self.transform.train_transfrom(trans_img), self.transform.train_transfrom(trans_img)


class CIFAR100SSL(datasets.CIFAR100):

    def __init__(self,
                 root,
                 indexs=None,
                 train=True,
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid

        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        self.transform = tranformSSL(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if not self.train or self.is_valid:
            return self.transform.test_transform(img), one_hot(target, 100)

        trans_img = self.weak(img)

        if self.is_labeled:
            return self.transform.train_transfrom(trans_img), one_hot(target, 100)

        return self.transform.train_transfrom(trans_img), self.transform.train_transfrom(trans_img)



class SVHNSSL(datasets.SVHN):

    def __init__(self,
                 root,
                 indexs=None,
                 split="train",
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, split=split, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.261, 0.2687]
        self.transform = tranformSSL(mean, std)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = img.transpose(1, 2, 0)

        if not self.train or self.is_valid:
            return self.transform.test_transform(img), one_hot(target, 10)

        trans_img = self.weak(img)

        if self.is_labeled:
            return self.transform.train_transfrom(trans_img), one_hot(target, 10)

        return self.transform.train_transfrom(trans_img), self.transform.train_transfrom(trans_img)