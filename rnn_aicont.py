# -*- coding: utf-8 -*-
import os
import sys
import time
import joblib
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from logger import setup_logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import torch 
import torch.nn as nn
import torch.optim as optim
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary


# training params
epochs=20
learning_rate=1e-4
batch_size=512
device = 'cuda' if torch.cuda.is_available else 'cpu'


class CustomDataset(Dataset):
    
    def __init__(self, root='/home/JinK/PyTorch/AIconnect/data', result_dir='/home/JinK/PyTorch/AIconnect/result', mode='train', transform=None):
        
        self.mode = mode
        self.result_dir = result_dir
        self.data_path = os.path.join(root, f'{mode}.csv')
        df = pd.read_csv(self.data_path, encoding='utf-8')
        
        if mode == 'train':
            features = df.iloc[:, 1:-3].copy().fillna(0)
            targets = df.iloc[:, -3:].copy().fillna(0)
        else:
            features = df.iloc[:, 1:].copy().fillna(0)
            
        # feature scaling
        feature_scaler = MinMaxScaler()
        feature_scaler_path = os.path.join(result_dir, 'feature_scaler.pkl')

        if mode == 'train':
            feature_scaler.fit(features)
            joblib.dump(feature_scaler, feature_scaler_path)
        else:
            feature_scaler = joblib.load(feature_scaler_path)

        self.features = feature_scaler.transform(features)
        
        # target encoding
        if mode != 'test':
            self.targets = targets.replace({'정상': 0,'주의': 1,'경고': 2})
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        
        if idx <= 5:
            temp = self.features[idx].reshape(1, -1)
            feature = np.concatenate([temp, temp, temp, temp, temp], axis=0)
            feature = torch.tensor(feature, dtype=torch.float)
        else:
            feature = self.features[idx-5:idx]
            feature = torch.tensor(feature, dtype=torch.float)

        if self.mode != 'test':
            target_1 = int(self.targets.iloc[idx, 0])
            target_2 = int(self.targets.iloc[idx, 1])
            target_3 = int(self.targets.iloc[idx, 2])

            target_1 = torch.tensor(target_1, dtype=torch.long)
            target_2 = torch.tensor(target_2, dtype=torch.long)
            target_3 = torch.tensor(target_3, dtype=torch.long)

            return (feature, target_1, target_2, target_3)

        else:
            return feature

train_set = CustomDataset(mode='train')

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

class rnn(nn.Module):
    
    def __init__(self, input_features, hidden_size, num_layers, num_classes=3):
        super(rnn, self).__init__()
        self.relu = nn.ReLU()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm 
        self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # gru
        self.gru1 = nn.GRU(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gru2 = nn.GRU(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gru3 = nn.GRU(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        
        self.bn1_1 = nn.BatchNorm1d(hidden_size)
        self.bn1_2 = nn.BatchNorm1d(hidden_size)
        self.bn1_3 = nn.BatchNorm1d(hidden_size)
        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)
        self.dp = nn.Dropout(0.2)
        
        # Dense1
        self.dense1 = nn.utils.weight_norm(nn.Linear(hidden_size, 32))
        self.dense1_2 = nn.utils.weight_norm(nn.Linear(32, 3))
        # Dense2
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, 32))
        self.dense2_2 = nn.utils.weight_norm(nn.Linear(32, 3))
        # Dense3
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, 32))
        self.dense3_2 = nn.utils.weight_norm(nn.Linear(32, 3))
    
    def forward(self, x):
        
        x_1, _ = self.gru1(x)
        ht = x_1[:, -1, :]
        x_1 = self.bn1_1(ht)
        x_1 = self.dp(x_1)

        x_1 = self.dense1(x_1)
        x_1 = self.bn2_1(x_1)
        x_1 = self.dp(x_1)
        x_1 = self.dense1_2(x_1)
        
        x_2, _ = self.gru2(x)
        ht = x_2[:, -1, :]
        x_2 = self.bn1_2(ht)
        x_2 = self.dp(x_2)
        
        x_2 = self.dense2(x_2)
        x_2 = self.bn2_2(x_2)
        x_2 = self.dp(x_2)
        x_2 = self.dense2_2(x_2)
        
        x_3, _ = self.gru3(x)
        ht = x_3[:, -1, :]
        x_3 = self.bn1_3(ht)
        x_3 = self.dp(x_3)
        
        x_3 = self.dense3(x_3)
        x_3 = self.bn2_3(x_3)
        x_3 = self.dp(x_3)
        x_3 = self.dense3_2(x_3)
        
        return x_1, x_2, x_3

model = rnn(23, 128, 1).to(device)

# # Resume training by loading latest weights of the model
# model.load_state_dict(torch.load(f'models/{model.__class__.__name__}-e50.pth'))

# set logger
logger = setup_logger("aiconnect_classification", '/home/JinK/PyTorch/AIconnect/runs/logs', 0,
                      filename='{}_train_log.txt'.format(model.__class__.__name__), mode='a+')

criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)
criterion3 = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

metric_fn = f1_score


for ep in range(epochs):
    
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    start_time = time.time()
    
    with tqdm(total=len(train_loader.dataset)) as progress_bar:
        for i, (features, t1, t2, t3) in enumerate(train_loader):
            
            features, t1, t2, t3 = features.to(device), t1.to(device), t2.to(device), t3.to(device)
            
            optimizer.zero_grad()
            
            o1, o2, o3 = model(features)
            
            l1 = criterion1(o1,t1)
            l2 = criterion2(o2,t2)
            l3 = criterion3(o3,t3)
            
            o1 = torch.argmax(o1, dim=1).cpu()
            o2 = torch.argmax(o2, dim=1).cpu()
            o3 = torch.argmax(o3, dim=1).cpu()
            
            t1, t2, t3 = t1.cpu(), t2.cpu(), t3.cpu()
            
            acc1 = metric_fn(t1, o1, average='macro')
            acc2 = metric_fn(t2, o2, average='macro')
            acc3 = metric_fn(t2, o3, average='macro')
            
#             bat_loss = (l1+l2+l3)
#             bat_loss.backward()
            l1.backward()
            l2.backward()
            l3.backward()
            
            optimizer.step()
            scheduler.step()
            
            train_loss += (l1+l2+l3)
            avg_loss = train_loss/(i+1)
            train_acc += (acc1+acc2+acc3)/3
            avg_acc = train_acc/(i+1)
            
            progress_bar.set_postfix(loss=avg_loss)
            progress_bar.update(features.size(0))
            
    print(f' Epoch : {ep+1} | Train Loss : {avg_loss:.5f} | Train Acc : {avg_acc:.4f}')
    
    logger.info("Epoch : {:d} | Lr: {:.6f} | Loss: {:.5f} | Cost Time: {}".format(
                    (ep+1), optimizer.param_groups[0]['lr'], avg_loss,
                    str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    
    if (ep+1) % 5 == 0:
        torch.save(model.state_dict(), f'models/{model.__class__.__name__}-e{ep+1}.pth')

torch.cuda.empty_cache()
