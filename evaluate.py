import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()
        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()
        

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP

def topk_precision(query_code,
                     database_code,
                     query_labels,
                     database_labels,
                     topk=None,
                     ):

    num_query = query_labels.shape[0]
    topk_precision = []

    for topkv in range(1, topk+1):

        precision_query = 0

        for i in range(num_query):
            # Retrieve images from database
            relevant = np.matmul(query_labels[i, :], database_labels.transpose()) > 0

            # Calculate hamming distance
            hamming_dist = 0.5 * (database_code.shape[1] - np.matmul(query_code[i, :], database_code.transpose()))

            # Arrange position according to hamming distance
            retrieval_relevant = relevant[np.argsort(hamming_dist)][:topkv]

            # Retrieval count
            retrieval_relevant_cnt = retrieval_relevant.sum()

            precision_query += retrieval_relevant_cnt

        precision_query = precision_query / num_query / topkv
        topk_precision.append(precision_query)

    return topk_precision


def topk_recall(query_code,
                     database_code,
                     query_labels,
                     database_labels,
                     topk=None,
                     ):

    num_query = query_labels.shape[0]
    topk_precision = []

    for topkv in range(1, topk+1):

        precision_query = 0

        for i in range(num_query):
            # Retrieve images from database
            relevant = np.matmul(query_labels[i, :], database_labels.transpose()) > 0

            # 真实retrieval的数量

            retrieval_relevant_cnt_true = retrieval_relevant.sum()

            # Calculate hamming distance
            hamming_dist = 0.5 * (database_code.shape[1] - np.matmul(query_code[i, :], database_code.transpose()))

            # Arrange position according to hamming distance
            retrieval_relevant = relevant[np.argsort(hamming_dist)][:topkv]

            # Retrieval count
            retrieval_relevant_cnt = retrieval_relevant.sum()

            precision_query += retrieval_relevant_cnt

            recall_query = precision_query/ retrieval_relevant_cnt_true

        precision_query = precision_query / num_query 
        topk_precision.append(precision_query)

    return topk_precision


def precision_recall(query_code,
                    database_code,
                    query_labels,
                    database_labels,
                    step=100,
                    topk=500
                    ):

    num_query = query_labels.shape[0]
    num_database = topk
    precision = []
    recall = []

    for topkv in range(1, (num_database+1), step):
        precision.append([])
        recall.append([])

    for i in range(num_query):
        # Retrieve images from database
        relevant = np.matmul(query_labels[i, :], database_labels.transpose()) > 0
        relevant_cnt = np.sum(relevant)
        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - np.matmul(query_code[i, :], database_code.transpose()))
        relevant_sort = relevant[np.argsort(hamming_dist)]

        for index, topkv in enumerate(range(1, (num_database+1), step)):
            # Arrange position according to hamming distance
            retrieval_relevant = relevant_sort[:topkv]

            # Retrieval count
            retrieval_relevant_cnt = retrieval_relevant.sum()
            if relevant_cnt == 0:
                continue
            else:
                recall[index].append(retrieval_relevant_cnt / relevant_cnt)
                precision[index].append(retrieval_relevant_cnt / topkv)
    for index, k in enumerate(range(1, (num_database+1), step)):
        # assert len(recall[index]) == num_query
        precision[index] = sum(precision[index]) / len(precision[index])
        recall[index] = sum(recall[index]) / len(recall[index])
    return precision, recall


def plot_precision_recall(df, path, dataset, bits, method_list, step):

    #####
    default_cycler = (cycler(color=['g', 'r', 'b','c','m']) +
                      cycler(linestyle=['-', '-', '-', '-','-']))

    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    plt.title(dataset.upper())
    ft_size = 20
    for i in range(len(method_list)):
        line1, = ax.plot(df["method{}_recall".format(i)], df["method{}_precision".format(i)], label=method_list[i])
    # ax.set_title("PR curves on {} with {} bits code".format(dataset, bits), fontsize=ft_size)
        # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top':
            spine.set_visible(False)
    ax.set_ylabel("Precision", fontsize=ft_size)
    ax.set_xlabel("Recall", fontsize=ft_size)

    # ax.legend()
    # plt.show()
    # plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left = 0.25)
    plt.savefig("{}/precision_recall_{}_{}_{}.png".format(path, dataset, bits, step))
    plt.close('all')
    plt.pause(0.02)
    plt.clf()


def plot_topk_precision(df, path, dataset, bits, method_list, step):

    #####
    default_cycler = (cycler(color=['g', 'r', 'b','c','m']) +
                      cycler(linestyle=['-', '-', '-', '-','-']))

    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    plt.title(dataset.upper())
    ft_size = 20
    for i in range(len(method_list)):
        ax.plot(df["k"], df["method{}_precision".format(i)], label=method_list[i])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top':
            spine.set_visible(False)
    ax.set_xlabel("Number of Top Ranked Samples", fontsize=ft_size)
    ax.set_ylabel("Precision", fontsize=ft_size)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left = 0.25)
    # plt.grid(linestyle='--', linewidth=0.5)

    # ax.legend()
    # plt.show()

    plt.savefig("{}/topk_precision_{}_{}_{}.png".format(path, dataset, bits, step))
    plt.close('all')
    plt.pause(0.02)
    plt.clf()

def plot_topk_recall(df, path, dataset, bits, method_list, step):

    #####
    default_cycler = (cycler(color=['g', 'r', 'b','c','m']) +
                      cycler(linestyle=['-', '-', '-', '-','-']))

    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    plt.title(dataset.upper())
    ft_size = 20
    for i in range(len(method_list)):
        ax.plot(df["k"], df["method{}_recall".format(i)], label=method_list[i])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top':
            spine.set_visible(False)
    ax.set_xlabel("Number of Top Ranked Samples", fontsize=ft_size)
    ax.set_ylabel("Recall", fontsize=ft_size)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left = 0.25)
    # plt.grid(linestyle='--', linewidth=0.5)

    # ax.legend()
    # plt.show()

    plt.savefig("{}/topk_recall_{}_{}_{}.png".format(path, dataset, bits, step))
    plt.close('all')
    plt.pause(0.02)
    plt.clf()

def topN(query_code,
        database_code,
        query_labels,
        database_labels,
        i,
        topk=None,
        ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """

 
    # Retrieve images from database
    retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

    # Calculate hamming distance
    hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

    # Arrange position according to hamming distance
    index = torch.argsort(hamming_dist)[:topk].numpy()

    retrieval = retrieval[torch.argsort(hamming_dist)[:topk]].cpu().numpy()

        
    return index, retrieval