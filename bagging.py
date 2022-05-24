import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest ,mutual_info_regression
from sklearn.feature_selection import chi2
import plott
from scipy.stats import linregress

import time
import torch
from torchvision import datasets, transforms

from torchensemble.voting import NeuralForestClassifier
from torchensemble.utils.logging import set_logger
from sklearn.model_selection import train_test_split
from Data_Preprocessing import GetCleanData
from torch.utils.data import TensorDataset, DataLoader

from torchensemble.fusion import FusionRegressor
from torchensemble.voting import VotingRegressor
from torchensemble.bagging import BaggingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.snapshot_ensemble import SnapshotEnsembleRegressor
import torch.nn as nn
from torch.nn import functional as F
from torchensemble.utils.logging import set_logger

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Read the csv file
data3 = pd.read_excel('Data/3_dependant_GenetikaExcercise_Confidential.xlsx')
data3 = pd.get_dummies(data3)
np_data = np.array(data3.values)


## Mean each patient
meanID = data3.groupby('ID').mean().reset_index()
data2 = pd.read_excel('Data/2_symptoms_GenetikaExercise_Confidential.xlsx')
data2 = pd.get_dummies(data2)

np_data2 = np.array(data2.values)
index_delete = []
for index,val in enumerate(np_data2):
    if val[0] not in np_data[:,0]:
        index_delete.append(index)
        print(index)
data2 = data2.drop(index_delete)
np_data2 = np.delete(np_data2,index_delete,0)
index_delete = []
for index,val in enumerate(np_data):
    if val[0] not in np_data2[:,0]:
        index_delete.append(index)
        print([index , val[0]])
np_data = np.delete(np_data,index_delete,0)
data3 = data3.drop(index_delete)


slopes = []
counter = 0
groups = data3.groupby('ID')
for group in groups:
    bool_clinical_dep = False
    for index,depression in enumerate(group[1].values[:, 2:].T[0]):
        if depression > 50:
            slopes.append(1)
            bool_clinical_dep = True
            break
    if not bool_clinical_dep:
        slopes.append(0)
    # week_diff = np.diff(group[1].values[:,1:2].T)
    # depression_diff = np.diff(group[1].values[:,2:].T)
    # if group[1].values[:,1:2].T[0].shape[0] >=2:
    #     slope, intercept, r, p, se = linregress(group[1].values[:,1:2].T[0], group[1].values[:,2:].T[0])
    #     slopes = np.insert(slopes,slopes.shape[0],slope)
    # else:
    #     slopes = np.insert(slopes,slopes.shape[0],group[1].values[:,2:].T[0])
    #     # print(group)
    #     counter +=1
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(48, batch_size)
        self.linear2 = nn.Linear(batch_size, batch_size)
        self.linear3 = nn.Linear(batch_size, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return

def display_records(records, logger):
    msg = (
        "{:<28} | Testing MSE: {:.2f} | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, mse in records:
        logger.info(msg.format(method, mse, training_time, evaluating_time))
if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 5
    depth = 5
    lamda = 1e-3
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 50

    # Utils
    # cuda = False
    n_jobs = 1
    batch_size = 64
    y = np.array(slopes).astype(int)
    records = []
    np_data2 = np.array(np_data2).astype(dtype='float32')

    # Converting numpy array to Tensor
    x_train_tensor = torch.from_numpy(np_data2).to(device)
    y_train_tensor = torch.from_numpy(y).to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        x_train_tensor, y_train_tensor, test_size=0.20, random_state=42)

    # Generators
    training_set = torch.utils.data.TensorDataset(X_train, y_train)
    # training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = torch.utils.data.TensorDataset(X_test, y_test)
    # validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    # Load data
    train_loader = torch.utils.data.DataLoader(training_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
    )

    logger = set_logger(
        "classification_mnist_tree_ensemble", use_tb_logger=False
    )


    # FusionRegressor
    model = FusionRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("FusionRegressor", training_time, evaluating_time, testing_mse)
    )

    # VotingRegressor
    model = VotingRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("VotingRegressor", training_time, evaluating_time, testing_mse)
    )

    # BaggingRegressor
    model = BaggingRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        ("BaggingRegressor", training_time, evaluating_time, testing_mse)
    )

    # GradientBoostingRegressor
    model = GradientBoostingRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(
        (
            "GradientBoostingRegressor",
            training_time,
            evaluating_time,
            testing_mse,
        )
    )

    # SnapshotEnsembleRegressor
    model = SnapshotEnsembleRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    display_records(records, logger)
