import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, max_error, r2_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils import weight_norm
from tqdm import tqdm
import os
import sys
import akshare as ak
from bayes_opt import BayesianOptimization
import talib

df = pd.read_csv('./ndx_data.csv')
# 特征衍生
df['MTM'] = df['close'] - df['close'].shift(5)
# 移动平均
df['MA_5'] = talib.MA(df['close'],timeperiod = 5)
#df['MA_10'] = talib.MA(df['close'],timeperiod = 10)
#df['MA_20'] = talib.MA(df['close'],timeperiod = 20)
# 指数移动平均
df['EMA'] = talib.EMA(df['close'],timeperiod = 26)
# 异同移动平均
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
# 相对强弱指标
df['RSI_6'] = talib.RSI(df['close'],timeperiod = 6)
#df['RSI_12'] = talib.RSI(df['close'],timeperiod = 12)
# 真实波幅
df['ATR'] = talib.ATR(df['high'],df['low'],df['close'],timeperiod= 12)
# 平均趋向指数
df['ADX'] = talib.ADX(df['high'],df['low'],df['close'], timeperiod= 14)
# 威廉指标
df['WR'] = talib.WILLR(df['high'],df['low'],df['close'],timeperiod=6)
# 顺势指标
df['CCI'] = talib.CCI(df['high'],df['low'],df['close'],timeperiod=14)
# 数据预处理
# 划分训练集和验证集
df = df.dropna()
X,y = df, df[['close']]

data_train_init,data_test_init = train_test_split(df,test_size=0.2,shuffle=False)
y_train_init,y_test_init = train_test_split(y,test_size=0.2,shuffle=False)
# print(data_train_init.shape,data_test_init.shape)
# 数据归一化
# 执行数据标准化
scaler_x = MinMaxScaler()
data_train_scaler = scaler_x.fit_transform(data_train_init)
data_test_scaler = scaler_x.transform(data_test_init)

scaler_y = MinMaxScaler()
y_train_scaler = scaler_y.fit_transform(y_train_init)
y_test_scaler = scaler_y.transform(y_test_init)

# 数据滑窗及划分特征与标签
def create_dataset(dataset,lookback = 30):
    '''
    实现数据滑窗操作：输出X,y（TenserFloat） 
    参数：
    X_data : 特征
    y_data : 标签
    lookback : 窗口长度 
    '''
    X,y = [],[]
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback,:]
        target = dataset[i+1:i+lookback+1,3]
        X.append(feature)
        y.append(target)
    return torch.FloatTensor(X).view(len(X),lookback,-1),torch.FloatTensor(y).view(len(X),lookback,1)

# 构建LSTM网络架构
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,device = torch.device('cuda')):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach().to(self.device), c0.detach().to(self.device)))
        out = self.fc(out[:, :, :])
        return out

# TCN网络架构
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block
        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size, dropout=0.2,is_output=True):
        """
        :param num_inputs: int， 输入通道数
        :param channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TCN, self).__init__()
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else channels[i - 1]  # 确定每一层的输入通道数
            out_channels = channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1],1)
        self.is_output = is_output
    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0, 2, 1)  # b i s 
        x = self.network(x)
        x = x.permute(0, 2, 1)
        if self.is_output:
            x = self.fc(x[:,:,:])
        return x
# TCN-LSTM网络架构
class TCN_LSTM(nn.Module):
    def __init__(self,num_inputs,channels,kernel_size,dropout,hidden_size,num_layers,output_dim):
        super(TCN_LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.tcn = TCN(num_inputs=self.num_inputs, channels=self.channels,kernel_size=self.kernel_size,dropout=self.dropout,is_output=False)
        self.lstm = nn.LSTM(input_size=self.channels[-1], hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_dim)

    def forward(self, x):
        x = self.tcn(x)  # b h s
        x, _ = self.lstm(x)  # b, s, h
        x = self.fc(x[:, :, :])  # b output_size
        return x
    
    
# 定义推理函数
def infer(model,dataset,device=torch.device('cuda')):
    '''
    推理函数:返回MSE,MAE,R_squra
    参数
    '''
    X_data,y_data = dataset[:][0],dataset[:][1]   
    model.eval()
    with torch.no_grad():
        y_pred = model(X_data.to(device)).cpu()

        y_data = np.array(y_data[:,9,:]).flatten()
        y_pred = np.array(y_pred[:,9,:]).flatten()
        
        # 计算均方误差 (MSE)
        mse = mean_squared_error(y_data, y_pred)   
        rmese = np.sqrt(mse)
        # 计算最大误差 (Max Error)
        mae = max_error(y_data, y_pred) 
        # 计算 R^2 (决定系数)
        r_squared = r2_score(y_data, y_pred)

    return rmese,mae,r_squared

# 定义训练函数
def fit(model, train_dataset, test_dataset, device, dim, learning_rate, n_epoch, batch_size):
    # 实例化模型
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    # 训练过程
    for epoch in range(n_epoch):
        model.train()
        loss_temp = []  # 用于存储每个批次的损失
        
        for feature, target in train_loader:
            optimizer.zero_grad()  # 清空梯度
            output = model(feature.to(device))  # 模型前向传播
            loss = loss_fn(output, target.to(device))  # 计算损失
            loss_temp.append(loss.item())  # 存储损失值
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        
        train_loss = np.mean(loss_temp)  # 计算平均损失
        
        if (epoch + 1) % 20 == 0:  # 每20个epoch输出一次
            print(f'Train epoch[{epoch + 1}/{n_epoch}] Train_Loss:{train_loss:.5f}')
        
    print('Finished training')

    # 训练集推断
    rmse, mae, rs = infer(model, train_dataset, device)
    print("Training set - RMSE:", rmse, "MAE:", mae, "R2:", rs)

    # 测试集推断
    rmse, mae, rs = infer(model, test_dataset, device)
    print("Test set - RMSE:", rmse, "MAE:", mae, "R2:", rs)
    return model


def get_pre(mdoel,data):
    mdoel.eval()
    
    pre = mdoel(data.to(device))
    pre = pre[:,9,:].cpu().detach().numpy()
    pre = scaler_y.inverse_transform(pre)
    
    return pre


# 训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 训练数据
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
dim = X_train.shape[-1]
print('dim=',dim)

# 设置模型参数
n_epoch = 60
batch_size = 128
learning_rate = 0.001

# TCN模型训练
tcn =TCN(num_inputs = dim,channels = [128,128,128],kernel_size=3,dropout=0.2)
tcn = fit(tcn,train_dataset,test_dataset,device,dim,learning_rate,n_epoch,batch_size)

# 模型训练LSTM
lstm = LSTM(input_dim= dim,
            hidden_dim=64,num_layers=2,output_dim=1)
lstm = fit(lstm,train_dataset,test_dataset,device,dim,learning_rate,n_epoch,batch_size)

# 模型训练TCN-LSTM
tcn_lstm = TCN_LSTM(num_inputs = dim
                    , channels = [32,32,32]
                    ,kernel_size = 3
                    ,dropout = 0.2
                    ,hidden_size = 64 
                    ,num_layers = 3
                    ,output_dim = 1)
tcn_lstm = fit(tcn_lstm,train_dataset,test_dataset,device,dim,learning_rate,n_epoch,batch_size)

# 贝叶斯优化
from bayes_opt import BayesianOptimization
def bayesopt_objective(learning_rate,channel,kernel_size,dropout,hidden_dim,num_layers):
    '''
    定义评估器
    '''
    n_epoch = 20
    batch_size = 128
    device = torch.device('cuda')
    num_inputs = dim
    channels = [int(channel)]*3
    
    tcn_lstm = TCN_LSTM(num_inputs = num_inputs
                    , channels = channels
                    ,kernel_size = int(kernel_size)+1
                    ,dropout = dropout
                    ,hidden_size = int(hidden_dim)
                    ,num_layers = int(num_layers)
                    ,output_dim = 1).to(device)
    optimizer = optim.Adam(tcn_lstm.parameters(), lr=learning_rate)
    # 定义损失函数
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    for epoch in range(n_epoch):
        tcn_lstm.train()
        loss_temp = []
        for feature,target in train_loader:
            optimizer.zero_grad()
            output = tcn_lstm(feature.to(device))
            loss = loss_fn(output,target.to(device))
            loss_temp.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(loss_temp)
    return -train_loss
# 定义参数空间
param_grid_simple = {'learning_rate':(0.001,0.01),
                     'channel':(16,128),
                     'kernel_size':(1,5),
                     'dropout':(0.0,0.5),
                     'hidden_dim':(16,128)
                     ,'num_layers':(1,5)
                    }
def param_bayes_opt(init_points, n_iter=10):
    opt = BayesianOptimization(bayesopt_objective
                              ,param_grid_simple
                              ,random_state=404)
    opt.maximize(init_points = init_points
                 ,n_iter = n_iter)
    params_best = opt.max['params']
    score_best = opt.max['target']
    print('\n','best params: ',params_best,
         '\n', 'best score: ',score_best)
    return params_best,score_best
params_best,score_best = param_bayes_opt(20,100)


bo_tcn_lstm = TCN_LSTM(num_inputs = dim
                    , channels = [56,56,56]
                    ,kernel_size = 3
                    ,dropout = 0.1414
                    ,hidden_size = 45
                    ,num_layers = 1
                    ,output_dim = 1).to(device)
bo_tcn_lstm = fit(tcn_lstm_bo,train_dataset,test_dataset,device,dim=dim,learning_rate = 0.00748,n_epoch =n_epoch,batch_size=128)

import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体（例如 SimHei）
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

plt.figure(figsize=(10,6),dpi=200)
tcn_pre = get_pre(tcn,X_test)
lstm_pre = get_pre(lstm,X_test)
tcn_lstm_pre = get_pre(tcn_lstm,X_test)
bo_tcn_lstm_pre = get_pre(bo_tcn_lstm,X_test)

plt.plot(tcn_pre,'y',label= 'TCN')
plt.plot(lstm_pre,'c',label='LSTM')
plt.plot(tcn_lstm_pre,'g',label='TCN-LSTM')
plt.plot(bo_tcn_lstm_pre,'r',label = 'BO-TCN-LSTM')
plt.plot(np.array(y_test_init[30:]),'k',label='真实值')

plt.xlim(left=0)
ax = plt.gca()
# 去除上边界和右边界
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('时间点', fontsize=12, labelpad=10)
plt.ylabel('收盘价', fontsize=12, labelpad=10)
plt.legend(loc='upper left', frameon=True, 
          facecolor='white', edgecolor='gray',
          fontsize=10)
# plt.savefig('prediction_comparison.png',  # 文件名
#             bbox_inches='tight',  # 去除白边
#             transparent=True,    # 背景透明可选
#             dpi=300)             # 印刷级分辨率

plt.show()