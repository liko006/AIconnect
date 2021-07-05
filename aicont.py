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
epochs=100
learning_rate=1e-4
batch_size=2048
device = 'cuda' if torch.cuda.is_available else 'cpu'


class CustomDataset(Dataset):
    
    def __init__(self, root='/home/JinK/PyTorch/AIconnect/data', result_dir='/home/JinK/PyTorch/AIconnect/result', mode='train', transform=None):
        
        self.mode = mode
        self.result_dir = result_dir
        self.data_path = os.path.join(root, f'{mode}.csv')
        df = pd.read_csv(self.data_path, encoding='utf-8')
        
        features = df.iloc[:, 1:-3].copy().fillna(0)
        targets = df.iloc[:, -3:].copy().fillna(0)
        
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
        self.targets = targets.replace({'정상': 0,'주의': 1,'경고': 2})
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
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

class fc_nn(nn.Module):
    
    def __init__(self, input_features, num_classes=3):
        super(fc_nn, self).__init__()
        self.relu = nn.ReLU()
        
        # Dense1
        self.batch_norm1 = nn.BatchNorm1d(input_features)
        self.batch_norm2 = nn.BatchNorm1d(input_features)
        self.batch_norm3 = nn.BatchNorm1d(input_features)
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(input_features, 128))
        self.dense2 = nn.utils.weight_norm(nn.Linear(input_features, 128))
        self.dense3 = nn.utils.weight_norm(nn.Linear(input_features, 128))
        
        # Dense2
        self.batch_norm1_2 = nn.BatchNorm1d(128)
        self.batch_norm2_2 = nn.BatchNorm1d(128)
        self.batch_norm3_2 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout(0.25)
        self.dense1_2 = nn.utils.weight_norm(nn.Linear(128, 256))
        self.dense2_2 = nn.utils.weight_norm(nn.Linear(128, 256))
        self.dense3_2 = nn.utils.weight_norm(nn.Linear(128, 256))
        
        # Dense3
        self.batch_norm1_3 = nn.BatchNorm1d(256)
        self.batch_norm2_3 = nn.BatchNorm1d(256)
        self.batch_norm3_3 = nn.BatchNorm1d(256)
        self.dropout_3 = nn.Dropout(0.4)
        self.dense1_3 = nn.utils.weight_norm(nn.Linear(256, 512))
        self.dense2_3 = nn.utils.weight_norm(nn.Linear(256, 512))
        self.dense3_3 = nn.utils.weight_norm(nn.Linear(256, 512))
        
        # Dense4
        self.batch_norm1_4 = nn.BatchNorm1d(512)
        self.batch_norm2_4 = nn.BatchNorm1d(512)
        self.batch_norm3_4 = nn.BatchNorm1d(512)
        self.dropout_4 = nn.Dropout(0.3)
        self.dense1_4 = nn.utils.weight_norm(nn.Linear(512, 128))
        self.dense2_4 = nn.utils.weight_norm(nn.Linear(512, 128))
        self.dense3_4 = nn.utils.weight_norm(nn.Linear(512, 128))
        
        # Dense4
        self.batch_norm1_5 = nn.BatchNorm1d(128)
        self.batch_norm2_5 = nn.BatchNorm1d(128)
        self.batch_norm3_5 = nn.BatchNorm1d(128)
        self.dropout_5 = nn.Dropout(0.2)
        self.dense1_5 = nn.utils.weight_norm(nn.Linear(128, 3))
        self.dense2_5 = nn.utils.weight_norm(nn.Linear(128, 3))
        self.dense3_5 = nn.utils.weight_norm(nn.Linear(128, 3))
    
    def forward(self, x):
        
        x_1 = self.batch_norm1(x)
        x_1 = self.dropout(x_1)
        x_1 = self.dense1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.batch_norm1_2(x_1)
        x_1 = self.dropout_2(x_1)
        x_1 = self.dense1_2(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.batch_norm1_3(x_1)
        x_1 = self.dropout_3(x_1)
        x_1 = self.dense1_3(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.batch_norm1_4(x_1)
        x_1 = self.dropout_4(x_1)
        x_1 = self.dense1_4(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.batch_norm1_5(x_1)
        x_1 = self.dropout_5(x_1)
        x_1 = self.dense1_5(x_1)
        
        x_2 = self.batch_norm2(x)
        x_2 = self.dropout(x_2)
        x_2 = self.dense2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.batch_norm2_2(x_2)
        x_2 = self.dropout_2(x_2)
        x_2 = self.dense2_2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.batch_norm2_3(x_2)
        x_2 = self.dropout_3(x_2)
        x_2 = self.dense2_3(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.batch_norm2_4(x_2)
        x_2 = self.dropout_4(x_2)
        x_2 = self.dense2_4(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.batch_norm2_5(x_2)
        x_2 = self.dropout_5(x_2)
        x_2 = self.dense2_5(x_2)
        
        x_3 = self.batch_norm3(x)
        x_3 = self.dropout(x_3)
        x_3 = self.dense3(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.batch_norm3_2(x_3)
        x_3 = self.dropout_2(x_3)
        x_3 = self.dense3_2(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.batch_norm3_3(x_3)
        x_3 = self.dropout_3(x_3)
        x_3 = self.dense3_3(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.batch_norm3_4(x_3)
        x_3 = self.dropout_4(x_3)
        x_3 = self.dense3_4(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.batch_norm3_5(x_3)
        x_3 = self.dropout_5(x_3)
        x_3 = self.dense3_5(x_3)
        
        return x_1, x_2, x_3
    
    
    
model = fc_nn(23).to(device)

# # Resume training by loading latest weights of the model
# model.load_state_dict(torch.load(f'models/{model.__class__.__name__}-e50.pth'))
    
# set logger
logger = setup_logger("aiconnect_classification", '/home/JinK/PyTorch/AIconnect/runs/logs', 0,
                      filename='{}_v2_train_log.txt'.format(model.__class__.__name__), mode='a+')

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
    
    logger.info("Epoch : {:d} | Acc: {:.4f} | Loss: {:.5f} | Cost Time: {}".format(
                    (ep+1), avg_acc, avg_loss,
                    str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    
    if (ep+1) % 5 == 0:
        torch.save(model.state_dict(), f'models/{model.__class__.__name__}_v2-e{ep+1}.pth')

torch.cuda.empty_cache()