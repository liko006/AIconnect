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
epochs=10
learning_rate=1e-4
batch_size=256
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

class cnn1d(nn.Module):
    
    def __init__(self, input_features, num_classes=3):
        super(cnn1d, self).__init__()
        self.relu = nn.ReLU()
        self.chan1 = 256
        self.chan2 = 512
        self.chan3 = 512
        self.input_features = input_features
        
        # Conv1
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(input_features, self.chan1, kernel_size=5, stride=1, padding=2))
        self.bn1 = nn.BatchNorm1d(self.chan1)
        self.dp1 = nn.Dropout(0.2)
        
        # Conv2
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(self.chan1, self.chan2, kernel_size=5, stride=1, padding=2))
        self.bn2 = nn.BatchNorm1d(self.chan2)
        self.dp2 = nn.Dropout(0.2)
        
        # Conv3
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(self.chan2, self.chan3, kernel_size=5, stride=1, padding=2))
        self.bn3 = nn.BatchNorm1d(self.chan3)
        self.dp3 = nn.Dropout(0.2)
        
        # Conv4
        self.conv4 = nn.utils.weight_norm(nn.Conv1d(self.chan3, self.chan1, kernel_size=5, stride=1, padding=2))
        self.bn4 = nn.BatchNorm1d(self.chan1)
        self.dp4 = nn.Dropout(0.2)
        
        # MaxPooling
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=2, padding=1, dilation=1)
        
        # Avg-Pool
        self.avepool1 = nn.AdaptiveAvgPool1d(output_size = self.chan2)
        self.avepool2 = nn.AdaptiveAvgPool1d(output_size = self.chan1)
        
        # Flatten
        self.flt = nn.Flatten()
        
        # Dense1
        self.dense1 = nn.utils.weight_norm(nn.Linear(self.chan1*256, 3))
        
        # Dense2
        self.dense2 = nn.utils.weight_norm(nn.Linear(self.chan1*256, 3))
        
        # Dense3
        self.dense3 = nn.utils.weight_norm(nn.Linear(self.chan1*256, 3))
    
    def forward(self, x):
        
        x = x.reshape(-1, self.input_features, 3)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = self.relu(x)
        x = self.avepool1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dp3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.dp4(x)
        x = self.relu(x)
        x = self.avepool2(x)
        
        x = self.flt(x)
        x_1 = self.dense1(x)
        x_2 = self.dense2(x)
        x_3 = self.dense3(x)
        
        return x_1, x_2, x_3

    

model = cnn1d(23).to(device)

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
            
            bat_loss = (l1+l2+l3)
            bat_loss.backward()
#             l1.backward()
#             l2.backward()
#             l3.backward()
            
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
