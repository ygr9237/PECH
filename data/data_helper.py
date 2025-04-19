import torch.utils.data as data

class AdvDataset(data.Dataset):
    # adv_data.shape = B X H X W X C
    # Image tensor data get normailized already
    def __init__(self, adv_data, labels):
        self.data = adv_data
        self.labels = labels

    def __getitem__(self, index):
        img = self.data[index]
        img_aug = img
        return img, img_aug, self.labels[index], index

    def __len__(self):
        return len(self.data)