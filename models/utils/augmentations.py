from PIL import ImageFilter
import random
from torchvision import transforms


dataset_attr = {
    'seq-cifar10': {
        'normalization': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        'size': 32,
    },
    'noisy-seq-cifar10': {
        'normalization': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        'size': 32,
    },
    'seq-cifar100': {
        'normalization': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        'size': 32,
    },
    'seq-tinyimg': {
        'normalization': transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),
        'size': 64,
    }
}


def norm_mean_std(size):
    if size == 28:  # CIFAR10, CIFAR100
        normalize = transforms.Normalize((0.13062755,), (0.30810780,))
    elif size == 32:  # CIFAR10, CIFAR100
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif size == 64:  # Tiny-ImageNet
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif size == 96:  # STL10
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:  # ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


class GaussianBlur(object):
    """Gaussian blur augmentation """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_color_distortion(s=1.0):
    """
    Color jitter from SimCLR paper
    @param s: is the strength of color distortion.
    """

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class SimCLRTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        if size == 224:  # ImageNet
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=size),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    # transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    # get_color_distortion(s=1.0),
                    # transforms.ToTensor(),
                    normalize
                ]
            )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SupContTransforms:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, dataset, apply_color_jitter=True, apply_gray_scale=True, apply_gaussian_blur=True):

        normalize = dataset_attr[dataset]['normalization']
        size = dataset_attr[dataset]['size']

        lst_transforms = [
            transforms.Resize(size=(size, size)),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
        ]
        if apply_color_jitter:
            lst_transforms.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        if apply_gray_scale:
            lst_transforms.append(transforms.RandomGrayscale(p=0.2))
        if apply_gaussian_blur:
            lst_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0.5 if size > 32 else 0.0))

        #lst_transforms = lst_transforms + [transforms.ToTensor(), normalize]
        lst_transforms = lst_transforms + [normalize]
        self.transform = transforms.Compose(lst_transforms)

    def __call__(self, x):
        return self.transform(x), self.transform(x)

