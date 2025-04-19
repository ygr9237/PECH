import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

         # if no centroids, by default just usual weight
        codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        
        return logits
def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
def load_model(arch, code_length, num_cluster,source_class):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length,num_cluster,source_class)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = model.classifier[:-3]
        # for i,param in enumerate(model.features.parameters()):
        #     if i <=2:
        #         param.requires_grad = False
        # model.classifier[1] = nn.ReLU(inplace=False)

        model = ModelWrapper(model, 4096, code_length,num_cluster,source_class)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.
    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length, num_cluster,source_class):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        # Extract features
        self.extract_features = False
        self.ce_fc = CosSim(code_length, num_cluster, learn_cent=False)
        self.emb = nn.Linear(last_node, source_class)

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        feature = self.model(x)
        y = self.hash_layer(feature)
        logit1 = self.ce_fc(y)
        logit2 = self.emb(feature)
        logit2 = nn.Softmax(dim=-1)(logit2)
        return logit1,logit2,feature, y
    
    def weight_norm(self):
        w = self.hash_layer.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.hash_layer.weight.data = w.div(norm.expand_as(w))

    def get_parameters(self):
        #parameter_list = [{"params":filter(lambda p: p.requires_grad,self.parameters()), "lr_mult":1, 'decay_mult':2}]
        # parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_extract_features(self, flag):
        """
        Extract features.
        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


