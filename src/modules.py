import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
from torchmetrics.functional import mean_absolute_percentage_error as mape_loss
import numpy as np
from torch.utils.data import DataLoader, Dataset
# The LikesDataset class is a PyTorch dataset that takes in a pandas DataFrame, a BERT tokenizer, and
# device information, and returns tokenized input, masks, counts, timestamps, and likes data for each
# item in the dataset.
class LikesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizerFast, device = torch.device('cuda')) -> None:
        super().__init__()
        self.data = data.to_numpy()
        self.titles = self.data[:, 0]
        self.counts = torch.tensor(self.data[:,  2: -4].astype(np.float64), device = device).float()
        self.times = []
        for i in data['created_at']:
            datetime_object = datetime.strptime(i, '%Y-%m-%d %H:%M:%S UTC')
            self.times.append([datetime_object.hour, datetime_object.minute, datetime_object.second])
        self.times = np.array(self.times)
        self.tree_base = torch.tensor(np.concatenate((self.times, self.data[:, -4: -1].astype(np.float64)), axis = 1), device = device).float()
        self.likes24 = torch.tensor(self.data[:,  -1].astype(np.float64), device = device).float()
        self.tokenizer = tokenizer
        self.max_len = 0 
        for ins in self.titles:
            self.max_len = max(len(ins), self.max_len)
        print(self.tree_base.shape, self.likes24.shape, self.counts.shape)
    def __getitem__(self, index):
        input_ids = self.tokenizer.encode(self.titles[index], return_tensors = 'pt')[0]
        if len(input_ids) < self.max_len:
            input_ids = torch.cat((input_ids, torch.zeros((self.max_len - len(input_ids)))))
        mask = (input_ids > 0).long()
        return input_ids.long(), mask.long(), torch.cat((self.counts[index], self.tree_base[index][:3]), dim = -1), self.likes24[index], self.tree_base[index]
    def __len__(self):
        return len(self.data)
        
# This is a PyTorch module for performing regression on text data using a BERT model and additional
# information.
class LikesRegression(nn.Module):
    def __init__(self, bert: BertModel, tokenizer: BertTokenizerFast, train_dataset = None, test_dataset = None, batch_size = 16, lr = 1e-4, lr_bert = 3e-5, embed_dim = 128, addition_dim = None):
        super().__init__()
        self.lr = lr
        self.lr_bert = lr_bert
        self.batch_size = batch_size
        self.dataset_train = train_dataset
        self.dataset_test = test_dataset
        self.bert = bert
        self.tokenizer = tokenizer
        
        self.addition_dim = addition_dim if train_dataset is None else len(train_dataset[0][2])
        self.embed_dim = embed_dim
        self.bert_embed = nn.Linear(in_features = self.bert.pooler.dense.out_features, out_features = self.embed_dim)
        self.addition_embed = nn.Linear(in_features = self.addition_dim, out_features = self.embed_dim)
        self.regresser = nn.Sequential( #nn.BatchNorm1d(num_features = self.concated_dim),
                                        nn.Linear(in_features = self.embed_dim * 2, out_features = self.embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(in_features = self.embed_dim, out_features = 128),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 128, out_features = 128),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(num_features = 128),
                                        nn.Linear(in_features = 128, out_features = 32),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 32, out_features = 1))
    def forward(self, text: torch.Tensor, addition: torch.Tensor, mask) -> torch.Tensor:
        # with torch.no_grad():
        logits = self.bert.forward(input_ids = text, attention_mask = mask).last_hidden_state[:, 0, :]
        concatenated = torch.cat((self.bert_embed(logits), self.addition_embed(addition)), dim = 1)
        # `out = self.regresser(concatenated)` is performing a regression on the concatenated tensor,
        # which is the concatenation of the output of the BERT model and the additional information
        # tensor. The `self.regresser` is a sequential neural network that takes the concatenated
        # tensor as input and outputs a single value, which is the predicted number of likes for the
        # given input.
        out = self.regresser(concatenated)
        return out
    def step(self, batch, batch_idx, device):
        text, mask, addition, y, for_tree = batch
        y = y.to(device)
        out = self(text.to(device), addition.to(device), mask.to(device)).view(y.shape)
        mse = F.mse_loss(out, y)
        # with torch.no_grad():
        mae = F.l1_loss(out, y)
        mape = mape_loss(out, y)
        return mse, mae, mape, (out.detach().clone().cpu().numpy(), addition.detach().clone().cpu().numpy(), for_tree.detach().clone().cpu().numpy(), y.detach().clone().cpu().numpy())
    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.bert.parameters(), 'lr': self.lr_bert},
                                {'params': self.regresser.parameters(), 'lr': self.lr}])
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size = self.batch_size, shuffle = True)