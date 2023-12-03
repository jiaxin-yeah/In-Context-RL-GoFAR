#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config

from ray import train, tune
from ray.train import Checkpoint
from ray.air import session
from ray.tune.schedulers import ASHAScheduler


# In[ ]:


device = torch.device("cuda")


# In[ ]:


d = 2
cells = 10
N = 100000 # number of offline datasets
N_train = N * 4 // 5
n = 100 # context length
n_train = 4 * n // 5
trials = 20 * n // 5 # online regret trials for val/test


# In[ ]:


dir = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])


# In[ ]:


def generate_B(d=2, c=10, N=100000, n=100, expert_prop=0.5):
    dsets, actions, coefficients = [], [], []
    
    goals = [np.array([i, j]) for i in range(cells) for j in range(cells)]
    np.random.shuffle(goals)
    for g in goals:
        for i in range(N // 100):
            while True:
                s = np.array([0, 0])
                a = np.random.choice(2 * d + 1, n)

                X = np.zeros((n + 1, 4 * d + 2))
                for j in range(n):
                    X[j + 1, 0:d] = s
                    
                    if i % (N // 100) < expert_prop * N // 100:
                        if s[0] < g[0]:
                            a[j] = 0
                        elif s[0] > g[0]:
                            a[j] = 1
                        elif s[1] < g[1]:
                            a[j] = 2
                        elif s[1] > g[1]:
                            a[j] = 3
                        else:
                            a[j] = 4
                    
                    X[j + 1, d + a[j]] = 1
                    s = np.clip(s + dir[a[j]], 0, cells - 1)
                    X[j + 1, -d - 1:-1] = s
                    if np.array_equal(s, g):
                        X[j + 1, -1] = 1
                X[0, 0:2] = X[-1, -3:-1]

                if np.max(X[:, -1]) == 1:
                    dsets.append(X)
                    actions.append(a)
                    coefficients.append(X[1:, -1])
                    break
            
    return dsets, actions, coefficients, goals


# In[ ]:


class DarkRoomDataset(Dataset):
    def __init__(self, dsets, actions, coefficients):
        self.dsets = dsets
        self.actions = actions
        self.coefficients = coefficients
    
    def __len__(self):
        return len(self.dsets)

    def __getitem__(self, idx):
        return self.dsets[idx], self.actions[idx], self.coefficients[idx]


# In[ ]:


class TransformerModel(nn.Module):
    def __init__(self, d=2, n_positions=n + 1, n_embd=32, n_layer=4, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self._read_in = nn.Linear(4 * d + 2, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 2 * d + 1)

    def forward(self, X):
        embeds = self._read_in(X)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        logit = self._read_out(output)
        return logit


# In[ ]:


def train_MDP(config):
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def loss_fn(pred, a, c): # weighted cross-entropy loss
        return torch.mean(ce_loss_fn(pred, a) * torch.exp(config["weight"] * c))

    model = TransformerModel()
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), config["lr"], weight_decay=config["weight_decay"])
    dsets, actions, coefficients, goals = generate_B()
    dsets_train = dsets[:N_train]
    actions_train = actions[:N_train]
    coefficients_train = coefficients[:N_train]
    goals_train, goals_val = goals[:n_train], goals[n_train:]

    train_data = DarkRoomDataset(dsets_train, actions_train, coefficients_train)
    train_dataloader = DataLoader(train_data, batch_size=6144)
    best_reward = 0
        
    for epoch in range(512):
        model.train() # training
        for batch, (X, a, c) in enumerate(train_dataloader):
            X = X.to(device).float()
            pred = torch.flatten(model(X)[:, :-1], 0, 1)

            a, c = a.flatten().to(device), c.flatten().to(device)
            loss = loss_fn(pred, a, c)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            X = torch.zeros((trials, 1, 4 * d + 2), device=device)
            for i in range(n + 1):
                policy = torch.softmax(model(X)[:, -1], 1).cpu().numpy()
                probabilities = np.random.rand(trials, 1)

                a = (probabilities < policy.cumsum(1)).argmax(1)
                X = X.cpu().numpy()
                s_next = np.clip(X[:, 0, :d] + dir[a], 0, cells - 1)

                X_next = np.zeros((trials, 1, 4 * d + 2))
                X_next[:, 0, :d] = X[:, 0, :d]
                X[:, 0, :d] = s_next
                X_next[np.arange(trials), 0, a + d] = 1
                X_next[:, 0, -d - 1:-1] = s_next
                X_next[:, 0, -1] = np.min(np.equal(s_next, np.repeat(np.array(goals_val), 20, 0)), 1).astype(float)
                X = torch.from_numpy(np.concatenate((X, X_next), 1)).type(torch.FloatTensor).to(device)
            reward = torch.mean(torch.sum(X[:, 1:, -1], 1)).item()
            best_reward = max(best_reward, reward)
            
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report({"best_return":best_reward, "return":reward},
                         checkpoint=Checkpoint.from_directory(tempdir))


# In[ ]:


config = {
    "lr": tune.loguniform(1e-6, 1e-1),
    "weight": tune.uniform(0, 20),
    "weight_decay": tune.loguniform(1e-3, 1),
}
scheduler = ASHAScheduler(
    metric="best_return",
    mode="max",
    max_t=512,
    reduction_factor=2,
)
result = tune.run(
    train_MDP,
    resources_per_trial={"cpu": 28, "gpu": 1},
    config=config,
    num_samples=50,
    scheduler=scheduler,
)


# In[ ]:


best_trial = result.get_best_trial("best_return", "max", "all")
best_checkpoint = result.get_best_checkpoint(best_trial, "return", "max")
print(f"Best trial config: {best_trial.config}")
print(f"Best checkpoint online regret: {best_trial.last_result['best_return']}")
print(f"Best checkpoint path: {best_checkpoint.path}")

best_model = TransformerModel()
best_model = nn.DataParallel(best_model)
best_model.to(device)

with best_checkpoint.as_directory() as checkpoint_dir:
    checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
    best_model.load_state_dict(checkpoint_dict["model_state_dict"])

