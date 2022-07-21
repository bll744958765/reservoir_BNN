# -*- coding: utf-8 -*-
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch
from torch import nn
import numpy as np
import pandas as pd
from torchsummary import summary
from math import sqrt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
def get_tensor_from_pd(dataframe_series):
    return torch.tensor(data=dataframe_series.values)


torch.__version__

seed = 42
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_num, hidden_num, bias=True),
            nn.Linear(hidden_num, hidden_num, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_num, hidden_num, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_num, hidden_num, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_num, hidden_num, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_num, hidden_num, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_num, output_num),

        )

    def forward(self, input):
        return self.net(input)

net = Net(input_num=14, hidden_num=256, output_num=2).to(device)
print(net)
summary(net, input_size=(14,))

scaler = StandardScaler()   #standand
# path1 = path of train data
# path2 = path of test data
# path3 = path of prediation
new_data = pd.read_csv('path1')
train_data = pd.read_csv('path2')
valid_data = pd.read_csv('path3')
train_data=np.array(train_data)
valid_data=np.array(valid_data)
train_x = train_data[:, 1:15]
train_y = train_data[:, 15]
valid1_x = valid_data[:, 1:15]
valid_y = valid_data[:, 15]
train_id = train_data[:,0]
valid_id = valid_data[:,0]
new1_data = np.array(new_data)
new_data = torch.FloatTensor(scaler.fit_transform(new1_data)).to(device).float()
train_y = torch.tensor(train_y).to(device).float().squeeze(-1)
valid_y = torch.tensor(valid_y).to(device).float().squeeze(-1)
train_x = torch.FloatTensor(scaler.transform(train_x)).to(device).float()
valid_x = torch.FloatTensor(scaler.transform(valid1_x)).to(device).float()

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# Uncertainty Loss
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, y,y_hat):
        si = torch.tensor(y_hat[:,1])
        sigma = torch.log(1+torch.exp(si))
        w = 0.5*(pow((y-y_hat[:,0]),2))
        loss = (torch.exp(-sigma)*w)+0.5*sigma
        return loss


learning_rate = 0.0005
batch_size =10
total_step = int(train_x.shape[0] / batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_func= UncertaintyLoss().to(device)

epochs =10
valid = []
prediction = []
train_sigma = []
lossData = []
valid_sigma = []
new_sigma = []

for m in range(2):
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    net.apply(weight_reset)
    epoch_train_loss_value = []
    step_train_loss_value = []
    step_valid_loss_value = []
    epoch_valid_loss_value = []
    step_train_sigma=[]
    epoch_train_sigma = []
    step_valid_sigma=[]
    epoch_valid_sigma = []
    for epoch in range(epochs):
        net.train()
        for step in range(total_step):
            xs = train_x[step * batch_size:(step + 1) * batch_size, :]
            ys = train_y[step * batch_size:(step + 1) * batch_size]
            pred = net(xs)
            loss = loss_func(ys,pred)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            step_train_loss_value.append(loss.cpu().detach().numpy())

        epoch_train_loss_value.append(np.mean(step_train_loss_value))
        loss_value = pd.DataFrame({'m': m,'epoch_train_loss_value':epoch_train_loss_value})

        if (epoch+1) % 5 == 0:
            print('m={:1d},epoch={:3d}/{:3d}, lr={:.4f},train_loss={:.4f}'.format(m+1,epoch + 1,
                 epochs,learning_rate,np.mean(step_train_loss_value).item()))

    predict = []
    sigma1=[]
    for i in range(train_x.shape[0]):
        y=net(train_x[i, :])
        predict.append(y[0].item())
        sigma1.append(y[1].item())
    prediction.append(predict)
    train_sigma.append(sigma1)


    for epoch in range(epochs):
        net.eval()
        enable_dropout(net)
        valid_loss = loss_func( valid_y,net(valid_x))
        epoch_valid_loss_value.append(valid_loss.cpu().detach().numpy())
        if (epoch+1) % 50 == 0:
            print('m={:1d},epoch={:3d}/{:3d}, lr={:.4f}, valid_loss={:.4f}'.format(m+1,epoch + 1,
                 epochs,learning_rate, np.mean(epoch_valid_loss_value).item(),))
    validate = []
    sigma2=[]
    for i in range(valid_x.shape[0]):
        v=net(valid_x[i, :])
        sigma2.append(v[1].item())
        validate.append(v[0].item())
    valid.append(validate)
    valid_sigma.append(sigma2)

    torch.save(net.state_dict(), 'model.pth')
    #predict on prediction data set
    network_params=net
    network_params.load_state_dict(torch.load('model.pth'))
    new_Y= net(new_data).cpu().detach().numpy()
    lossData.append(new_Y[:,0])
    new_sigma.append(new_Y[:,1])

#evaluation on train data set
confidence_lower = []
confidence_upper = []
confidence_width = []
prediction = np.array(prediction)
train_sigma = np.array(train_sigma)
n = pow(len(prediction), -0.5)
prediction_mean = prediction.mean(axis=0)
prediction_sigma = train_sigma.mean(axis=0)/len(train_sigma)
prediction_var = (prediction**2).mean(axis=0)-(prediction_mean**2)
prediction_std = np.sqrt(prediction_var)
prediction_mean=np.array(prediction_mean)
confidence_lower.append((prediction_mean - 1.96 * n * prediction_std))
confidence_upper.append((prediction_mean + 1.96 * n * prediction_std))
confidence_upper = np.array(confidence_upper).T.squeeze(-1)
confidence_lower=np.array(confidence_lower).T.squeeze(-1)
prediction = pd.DataFrame({"prediction_mean": prediction_mean,'confidence_lower':
             np.array(confidence_lower),'confidence_upper': np.array(confidence_upper)})
prediction.to_csv('./train.csv',index=False)

train_sigma = torch.exp(torch.tensor(train_sigma))
train_y=train_y.cpu().detach().numpy()
mse = (np.sum((train_y - prediction_mean) ** 2)) / len(train_y)
rmse = np.sqrt(mse)
mae = (np.sum(np.absolute(train_y - prediction_mean))) / len(train_y)
r2 = 1-(mse/((train_y**2).mean()-(train_y.mean()**2)))

print(" MAE:",mae,"MSE:",mse," RMSE:",rmse," R-square:",r2)
print("aleatoric uncertainty：",np.array(train_sigma).mean())
print("epistemic uncertainty：",prediction_var.mean())
print("uncertainty：",np.array(train_sigma).mean()+prediction_var.mean())


# evaluation on test data set
valid = np.array(valid)
valid_sigma=np.array(valid_sigma)
Vconfidence_lower = []
Vconfidence_upper = []
Vconfidence_width = []
n = pow(len(valid), -0.5)
validation_mean = valid.mean(axis=0)
validation_sigma=valid_sigma.mean(axis=0)/len(valid_sigma)
validation_var = (valid**2).mean(axis=0)-(validation_mean**2)
validation_std = np.sqrt(validation_var)
Vconfidence_lower.append((validation_mean - 1.96 * n * validation_std))
Vconfidence_upper.append((validation_mean + 1.96 * n * validation_std))
Vconfidence_upper = np.array(Vconfidence_upper).T.squeeze(-1)
Vconfidence_lower = np.array(Vconfidence_lower).T.squeeze(-1)
Vconfidence_width = Vconfidence_upper-Vconfidence_lower

valid_y=valid_y.cpu().detach().numpy()
validation_mean=np.array(validation_mean)
validation_sigma=torch.tensor(validation_sigma)
validation_sigma=torch.exp(validation_sigma)
print('valid_y',valid_y.shape,type(train_y))
print('validation_mean',validation_mean.shape,type(validation_mean))
valid_mse = (np.sum((valid_y - validation_mean) ** 2)) / len(valid_y)
valid_rmse = sqrt(valid_mse)
valid_mae = (np.sum(np.absolute(valid_y - validation_mean))) / len(valid_y)
valid_var=(valid_y**2).mean()-((valid_y.mean())**2)
valid_r2 = 1-(valid_mse/ ((valid_y**2).mean()-((valid_y.mean())**2)))
print("valid_mae:",valid_mae,"valid_mse:",valid_mse," valid_rmse:",valid_rmse," r2:",valid_r2)
print("aleatoric uncertainty:",np.array(validation_sigma).mean())
print("epistemic uncertainty:",validation_var.mean())
print("uncertainty:",validation_var.mean()+np.array(validation_sigma).mean())

# prediction data set
Data = np.array(lossData)
new_sigma=np.array(new_sigma)
print('Data shape:',Data.shape)
Nconfidence_lower = []
Nconfidence_upper = []
Nconfidence_width = []
print('Data shape',Data.shape)
n = pow(len(Data), -0.5)
print('len(d):',len(Data))
N_sigma=new_sigma.mean(axis=0)/len(new_sigma)
Nprediction_mean = Data.mean(axis=0)
Nprediction_var = (Data**2).mean(axis=0)-(Nprediction_mean**2)
Nprediction_std = np.sqrt(Nprediction_var)
Nconfidence_lower.append((Nprediction_mean - 1.96 * n * Nprediction_std))
Nconfidence_upper.append((Nprediction_mean + 1.96 * n * Nprediction_std))
Nconfidence_upper = np.array(Nconfidence_upper).T.squeeze(-1)
Nconfidence_lower = np.array(Nconfidence_lower).T.squeeze(-1)
Nconfidence_width = Nconfidence_upper-Nconfidence_lower
N_sigma=torch.exp(torch.tensor(N_sigma))

print("aleatoric uncertainty:",np.array(N_sigma).mean())
print("epistemic uncertainty:",np.array(Nprediction_var).mean())
print("uncertainty:",np.array(N_sigma).mean()+np.array(Nprediction_var).mean())
