import numpy as np
import pandas as pd
import os

path = os.path.abspath(os.getcwd()).replace('\\','/')
# path = path[:path.rfind('/') + 1]
DATA_PATH = path + "data/"

train = pd.read_csv(DATA_PATH + 'train.csv')

ratings = train['rating'].values
userIds = train['userId'].values
itemIds = train['movieId'].values

from scipy.sparse import coo_matrix

n_users = np.max(userIds) + 1
n_items = np.max(itemIds) + 1

R = coo_matrix((ratings, (userIds, itemIds)), shape=(n_users, n_items))

n_features = 15

P = np.random.random((n_users,n_features))
Q = np.random.random((n_items,n_features))

lr = 160
l2 = 0.000001

#train loop
for e in range(250):
    P_tau = P[userIds,:]
    Q_tau = Q[itemIds,:]
    pred = np.sum(P_tau * Q_tau, axis = 1)#np.inner(P_tau, Q_tau)
    R_hat = coo_matrix((pred, (userIds, itemIds)), shape=(n_users, n_items))
    MSE = np.sum(np.square(ratings - pred))/ratings.shape[0] + l2 * (np.sum(np.square(Q)) + np.sum(np.square(P)))
    P -= lr * ((R_hat - R) @ Q)/ratings.shape[0] + l2 * np.square(P)
    Q -= lr * ((R_hat - R).T @ P)/ratings.shape[0] + l2 * np.square(Q)
    lr *= 0.998
    print(f"epoch {e}: MSE = {MSE}, lr = {lr}")
np.save(path + 'models\\P.arr', P, allow_pickle=True, fix_imports=True)
np.save(path + 'models\\Q.arr', Q, allow_pickle=True, fix_imports=True)
    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_sz = 128
n_samples = len(ratings)

batches = []

for i in range(0, n_samples, batch_sz):
    limit =  min(i + batch_sz, n_samples)
    users_batch, movies_batch, rates_batch = userIds[i: limit], itemIds[i: limit], ratings[i: limit]
    batches.append((torch.tensor(users_batch, dtype=torch.long), torch.tensor(movies_batch, dtype=torch.long),
                  torch.tensor(rates_batch, dtype=torch.float)))

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
net = RecommenderNet(n_factors = 20, n_users=n_users, n_movies=n_items).to(device)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)

epochs = 20

for epoch in range(epochs):
    train_loss = 0
    c = 0
    for users_batch, movies_batch, rates_batch in batches:
        net.zero_grad()
        out = net(users_batch.to(device), movies_batch.to(device), [1, 5]).squeeze()
        loss = criterion(rates_batch.to(device), out)

        loss.backward()
        optimizer.step()
        train_loss += loss
        
        c += 1
    scheduler.step(loss)
    print("Loss at epoch {} = {}".format(epoch, train_loss/c))
    
torch.save(net.state_dict(), path + 'models\\NN.t')
