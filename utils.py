from typing import Optional
from torch.optim.optimizer import Optimizer
import sys
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Function

import torch
from torchvision import models
#from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.hub import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 其他模型的URL...
}

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:
    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},
    where `i` is the iteration steps.
    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class ResizeImage(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class AccuracyCounter:

    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)


def get_consistency(y_1, y_2, y_3, y_4, y_5):
    y_1 = torch.unsqueeze(y_1, 1)
    y_2 = torch.unsqueeze(y_2, 1)
    y_3 = torch.unsqueeze(y_3, 1)
    y_4 = torch.unsqueeze(y_4, 1)
    y_5 = torch.unsqueeze(y_5, 1)
    c = torch.cat((y_1, y_2, y_3, y_4, y_5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency


def get_entropy(y_1, y_2, y_3, y_4, y_5):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
    entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
    entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
    entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))

    entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return entropy


def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    return entropy


def get_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, indice_1 = torch.max(y_1, 1)
    conf_2, indice_2 = torch.max(y_2, 1)
    conf_3, indice_3 = torch.max(y_3, 1)
    conf_4, indice_4 = torch.max(y_4, 1)
    conf_5, indice_5 = torch.max(y_5, 1)
    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence


# def norm(t):
#     mean = torch.mean(t)
#     var = torch.std(t) ** 2
#     t = (t - mean) / var
#     return t

def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x



class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        del self.fc

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class DomainDiscriminator(nn.Module):

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1.}]


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t, w_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        target_loss = torch.mean(w_t * self.bce(d_t, d_label_t).view(-1))
        return 0.5 * (target_loss)


class ClassifierBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)


class Ensemble(nn.Module):

    def __init__(self, in_feature, num_classes):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(in_feature, num_classes)
        self.fc2 = nn.Linear(in_feature, num_classes)
        self.fc3 = nn.Linear(in_feature, num_classes)
        self.fc4 = nn.Linear(in_feature, num_classes)
        self.fc5 = nn.Linear(in_feature, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x, index=0):
        if index == 1:
            y = self.fc1(x)
            # y = nn.Softmax(dim=-1)(y_1)
        elif index == 2:
            y = self.fc2(x)
            # y = nn.Softmax(dim=-1)(y_2)
        elif index == 3:
            y = self.fc3(x)
            # y = nn.Softmax(dim=-1)(y_3)
        elif index == 4:
            y = self.fc4(x)
            # y = nn.Softmax(dim=-1)(y_4)
        elif index == 5:
            y = self.fc5(x)
            # y = nn.Softmax(dim=-1)(y_5)
        else:
            y_1 = self.fc1(x)
            y_1 = nn.Softmax(dim=-1)(y_1)
            y_2 = self.fc2(x)
            y_2 = nn.Softmax(dim=-1)(y_2)
            y_3 = self.fc3(x)
            y_3 = nn.Softmax(dim=-1)(y_3)
            y_4 = self.fc4(x)
            y_4 = nn.Softmax(dim=-1)(y_4)
            y_5 = self.fc5(x)
            y_5 = nn.Softmax(dim=-1)(y_5)
            return y_1, y_2, y_3, y_4, y_5

        return y

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.parameters(), "lr_mult": 1.},
        ]
        return params
