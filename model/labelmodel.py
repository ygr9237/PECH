import torch
from torch import nn
from torch.nn import functional as F
def weights_init(m):
    nn.init.xavier_uniform(m.weight.data)
    nn.init.constant(m.bias.data, 0.00)

class Label_net(nn.Module):
    def __init__(self, label_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(Label_net, self).__init__()
        self.module_name = "text_model"
        # 400
        cl1 = nn.Linear(label_dim, 512)
        # cl3 = nn.Linear(4096, 512)
        cl2 = nn.Linear(512, bit)
        # print(cl2.weight.data)
        # weights_init(cl1)
        # weights_init(cl2)
        # # weights_init(cl3)
        # print(cl2.weight.data)
        self.cl_text = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            # cl3,
            # nn.ReLU(inplace=True),
            cl2,
            nn.Tanh()
        )
    def forward(self, x):
        y = self.cl_text(x)
        return y