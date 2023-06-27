# When we make a new one, we should inherit the Finetune class.
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import get_statistics
from utils.train_utils import select_model, select_optimizer
from utils.train_utils import ModelWrapper
from utils.data_loader import ImageDataset

from torch.utils.data import DataLoader

from methods.er_new import ER

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class REMIND(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)

        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        extract_features_from='model.layer4.0'

        model_F = select_model(self.model_name, self.dataset, 1, F=True)
        self.model_F = model_F.to(device)
        model_G = select_model(self.model_name, self.dataset, 1, G=True)
        model_G = ModelWrapper(model_G, output_layer_names=[extract_features_from], return_single=True)

        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model_F)
        self.train_datalist2 = train_datalist


    def online_step(self, sample, sample_num, n_worker):
        self.sample_num  = sample_num
        if sample_num==0:
            self.batch_init()
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_sample_nums.append(sample_num)
            
    def batch_init(self):
        self.model_G.to(self.device)
        self.model_G.eval()
        train_df = pd.DataFrame(self.train_datalist2)
        train_dataset = ImageDataset(train_df, self.dataset, self.train_transform, data_dir=self.data_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        start_ix = 0
        num_channels = self.model_G.in_features
        spatial_feat_dim = 7
        features_data = np.empty((len(train_loader.dataset), num_channels, spatial_feat_dim, spatial_feat_dim), dtype=np.float32)
        labels_data = np.empty((len(train_loader.dataset), 1), dtype=np.int)
        item_ixs_data = np.empty((len(train_loader.dataset), 1), dtype=np.int)
        for i, data in enumerate(train_loader):
            inputs = data['image']
            targets = data['label']
            batchs = inputs.to(self.device)
            # targets = targets.to(self.device)
            outputs = self.model_G(batchs)
            end_ix = start_ix + len(outputs)
            features_data[start_ix:end_ix] = outputs.cpu().numpy()
            labels_data[start_ix:end_ix] = np.atleast_2d(targets.numpy().astype(np.int)).transpose()
            item_ixs_data[start_ix:end_ix] = np.atleast_2d(i.numpy().astype(np.int)).transpose()
            start_ix = end_ix

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

