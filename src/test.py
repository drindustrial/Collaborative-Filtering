import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = os.path.abspath(os.getcwd()).replace('\\','/')
# path = path[:path.rfind('/') + 1]
DATA_PATH = path + "/data/"

test = pd.read_csv(DATA_PATH + 'test.csv')
train = pd.read_csv(DATA_PATH + 'train.csv')

ratings = train['rating'].values
userIds = train['userId'].values
itemIds = train['movieId'].values
n_users = np.max(userIds) + 1
n_items = np.max(itemIds) + 1

P = np.load(path + 'models\\P.arr.npy')
Q = np.load(path + 'models\\Q.arr.npy')

ratings_test = test['rating'].values
userIds_test = test['userId'].values
itemIds_test = test['movieId'].values
P_tau = P[userIds_test,:]
Q_tau = Q[itemIds_test,:]
pred = np.sum(P_tau * Q_tau, axis = 1)
MSE_test_MF = np.sum(np.square(ratings_test - pred))/ratings_test.shape[0]

class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, embedding_dropout=0.02, dropout_rate=0.2):
        super().__init__()

        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(nn.Linear(2*n_factors, n_factors*4),
                                nn.ReLU(),
                                nn.Dropout(0.15),
                                nn.Linear(n_factors*4, 2*n_factors),
                                nn.ReLU())
        self.fc = nn.Linear(n_factors*2, 1)
        self._init()

    def forward(self, users, movies, minmax=[1,5]):
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))

        if minmax is not None: #Scale the output to [1,5]
            min_rating, max_rating = minmax
            out = (max_rating - min_rating)*out + min_rating
        return out

    def _init(self):
        """
        Initialize embeddings and hidden layers weights with xavier.
        """
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)

model = RecommenderNet(n_factors = 20, n_users=n_users, n_movies=n_items).to(device)
model.load_state_dict(torch.load(path + 'models\\NN.t'))

ratings_test = test['rating'].values
userIds_test = test['userId'].values
itemIds_test = test['movieId'].values
pred = model.forward(torch.tensor(userIds_test).to(device),torch.tensor(itemIds_test).to(device)).cpu().detach().numpy()
pred = np.array([s[0] for s in pred])
MSE_test_nn = np.sum(np.square(ratings_test - pred))/ratings_test.shape[0]

print(f"basic Collaborative Filtering model MSE on test dataset = {MSE_test_MF}")
print(f"Deep learning model MSE on test dataset = {MSE_test_nn}")
