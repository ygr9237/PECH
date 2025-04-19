import torch
import torch.nn as nn
import torch.optim as optim
import clip
import numpy as np
import os
import time

import torchvision.transforms as transforms
from loguru import logger
import torch.nn.functional as F
from data.data_helper import AdvDataset
from torch.utils.data.dataset import ConcatDataset
from model.model_loader import load_model
from evaluate import mean_average_precision
from model.labelmodel import *
from torch.nn import Parameter
from torch.autograd import Variable
from utils import *
import random
from PIL import ImageFilter
from collections import OrderedDict

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)


def train(train_s_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          batch_size,
          knn,
          CLIP,
          classes,
          Domain_ID,
          n_classes,
          ):
    clip_model, _ = clip.load(CLIP, device=device)
    text_feature_dim = 512
    t0 = torch.cat([clip.tokenize(f"a photo of {c}.") for c in classes]).to(device)
    text_list = []
    source = Domain_ID.copy()
    text_anchor = source
    for source in text_anchor:
        text_list.append(torch.cat([clip.tokenize(f"a {source} of a {c}") for c in classes]).to(device))
    text_token_list = []
    with torch.no_grad():
        clip_model.eval()
        text0 = clip_model.encode_text(t0)

        for text in text_list:
            text_token_list.append(clip_model.encode_text(text))

        text_features_ems = text0

        text_features_ems /= text_features_ems.norm(dim=-1, keepdim=True)

    text_compare_teacher = torch.zeros(n_classes, len(text_anchor), text_feature_dim).to(device)

    for i in range(n_classes):
        for j in range(len(text_anchor)):
            text_compare_teacher[i, j, :] = text_token_list[j][i]

    model = load_model(arch, code_length, num_class, num_class)
    model.to(device)
    parameter_list = model.get_parameters()
    optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()

    max_iter_t = max_iter * len(train_s_dataloader)
    interval_iter = max_iter_t // 30
    iter_num = 0
    print(f"begin training:")
    print(f"max_iter_t: {max_iter_t}, interval_iter: {interval_iter}, iter_num: {iter_num}")

    for epoch in range(max_iter):
        for batch_idx, (data_s, _, target_s, index) in enumerate(
                train_s_dataloader):
            data_s = data_s.to(device)
            target_s = target_s.to(device)
            optimizer.zero_grad()
            logit_s, f_s, feature_s, code_s = model(data_s)
            loss_ce = criterion(logit_s, target_s.argmax(1))

            with torch.no_grad():
                clip_model.eval()
                CLIP_image_features = clip_model.encode_image(data_s)
            CLIP_image_features /= CLIP_image_features.norm(dim=-1, keepdim=True)

            teacher_logits = (100.0 * CLIP_image_features @ text_features_ems.t().mul(torch.tensor(3.0))).type(
                torch.float32)

            f_s_resized = f_s[:, :teacher_logits.shape[1]]   #维度不匹配
            kl_loss = F.kl_div(F.log_softmax(f_s_resized / 3.0, dim=1),
                               F.softmax(teacher_logits / 3.0, dim=1),
                               reduction='batchmean') * 3.0 * 3.0
            loss = loss_ce + kl_loss

            loss.backward(retain_graph=True)
            iter_num += 1
            optimizer.step()
            optimizer.zero_grad()

            if iter_num % interval_iter == 0 or iter_num == max_iter_t:
                log_str = 'Iter:{}/{}; loss:{:.4f}'.format(iter_num, max_iter_t, loss.item())
                print(log_str)

        if (epoch % evaluate_interval == evaluate_interval - 1):
            mAP = evaluate(model,
                           query_dataloader,
                           retrieval_dataloader,
                           code_length,
                           device,
                           topk,
                           save=True,
                           )
            logger.info('[Epoch:{}/{}][map:{:.4f}]'.format(
                epoch + 1,
                max_iter,
                mAP,
            ))

    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=False,
                   )
    logger.info('Training finish, [Epoch:{}][map:{:.4f}]'.format(epoch + 1, mAP))
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")



def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)

    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP),
                onehot_retrieval_targets.cpu().detach().numpy())

    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            _, _, _, outputs = model(data)
            code[index, :] = outputs.sign().cpu()

    return code

