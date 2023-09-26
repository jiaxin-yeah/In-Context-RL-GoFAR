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

device = torch.device("cuda:0")

A = 5 # number of actions
N = 80000 # number of offline datasets
n = 500 # context length
trials = 200 # online regret trials for val/test

def generate_B(A=5, N=80000, n=500):
    dsets, actions, coefficients = [], [], []
    for i in range(N):
        mu = np.random.rand(A)

        p_1 = np.random.dirichlet(np.ones(A))
        p_2 = np.zeros(A)
        p_2[np.random.choice(A)] = 1
        w = (np.random.choice(11)) / 10
        p = (1 - w) * p_1 + w * p_2
        
        a = np.random.choice(A, n, p=p)
        actions.append(a)
        
        r = np.random.normal(mu[a], 0.3)
        mu_empirical = np.zeros(A)
        for i in range(A):
            if np.argwhere(a==i).size > 0:
                mu_empirical[i] = r[np.where(a==i)].mean()
        coefficients.append(np.maximum(mu_empirical[a] - r.mean(), 0))
        
        X = np.zeros((n, A + 3), np.float32)
        X[:, [0, -2]] = 1
        X[np.arange(n), a + 1] = 1
        X[:, -1] = r
        dsets.append(X)
    return dsets, actions, coefficients

class BanditDataset(Dataset):
    def __init__(self, dsets, actions, coefficients):
        self.dsets = dsets
        self.actions = actions
        self.coefficients = coefficients
        
        self.first = np.zeros((1, A + 3), dtype=np.float32)
        self.first[0, 0] = 1
    
    def __len__(self):
        return len(self.dsets)

    def __getitem__(self, idx):
        perm = np.random.permutation(n) # shuffled in-context datset to reduce overfitting
        sample_ds = np.concatenate((self.first, self.dsets[idx][perm]))
        return sample_ds, self.actions[idx][perm], self.coefficients[idx][perm]

class TransformerModel(nn.Module):
    def __init__(self, n_states, n_positions=501, n_embd=32, n_layer=4, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_states
        self._read_in = nn.Linear(n_states + 3, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_states)

        for w in self._backbone.wpe.parameters(): # remove positional embedding
            w.data.fill_(0)
        self._backbone.wpe.weight.requires_grad=False

    def forward(self, X):
        embeds = self._read_in(X)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        logit = self._read_out(output)
        return logit

def train_bandit(config):
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    def loss_fn(pred, a, c): # weighted cross-entropy loss
        return torch.mean(ce_loss_fn(pred, a) * c)
    
    model = TransformerModel(n_states=5)
    model = nn.DataParallel(model)
    model.to(device)

    # do not train positional embedding
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad == True],
                                  config["lr"], weight_decay=config["weight_decay"])
    dsets_train, actions_train, coefficients_train = generate_B()
    
    train_data = BanditDataset(dsets_train, actions_train, coefficients_train)
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"])
    size = len(train_dataloader.dataset)
    best_online_regret = np.inf
    
    for epoch in range(256):
        model.train() # training
        for batch, (X, a, c) in enumerate(train_dataloader):
            X = X.to(device)
            pred = torch.flatten(model(X)[:, :-1], 0, 1)
            
            a, c = a.flatten().to(device), c.flatten().to(device)
            loss = loss_fn(pred, a, c)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        model.eval() # validation
        with torch.no_grad():
            online_regret = 0
            mu = np.random.rand(trials, A)

            # running all val trials in same tensor
            X = torch.zeros((trials, 1, A + 3), device=device)
            X[:, 0, 0] = 1
            for i in range(n + 1):
                policy = torch.softmax(model(X)[:, -1], 1).cpu().numpy()
                probabilities = np.random.rand(trials, 1)
                
                a = (probabilities < policy.cumsum(1)).argmax(1)
                r = np.random.normal(mu[np.arange(trials), a], 0.3)
                online_regret += (mu.max(1) - mu[np.arange(trials), a]).mean()
                a, r = torch.from_numpy(a).to(device), torch.from_numpy(r).to(device)
                
                X_new = torch.zeros((trials, 1, A + 3), device=device)
                X_new[:, 0, [0, -2]] = 1
                X_new[torch.arange(trials), 0, a + 1] = 1
                X_new[:, 0, -1] = r
                X = torch.torch.cat((X, X_new), 1)
            best_online_regret = min(best_online_regret, online_regret)

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report({"best_online_regret":best_online_regret, "online_regret":online_regret},
                         checkpoint=Checkpoint.from_directory(tempdir))

def online_regret(model):
    model.eval() # test
    with torch.no_grad():
        online_regret = 0
        mu = np.random.rand(trials, A)
        
        X = torch.zeros((trials, 1, A + 3), device=device)
        X[:, 0, 0] = 1
        for i in range(n + 1):
            policy = torch.softmax(model(X)[:, -1], 1).cpu().numpy()
            probabilities = np.random.rand(trials, 1)
            
            a = (probabilities < policy.cumsum(1)).argmax(1)
            r = np.random.normal(mu[np.arange(trials), a], 0.3)
            online_regret += (mu.max(1) - mu[np.arange(trials), a]).mean()
            a, r = torch.from_numpy(a).to(device), torch.from_numpy(r).to(device)
            
            X_new = torch.zeros((trials, 1, A + 3), device=device)
            X_new[:, 0, [0, -2]] = 1
            X_new[torch.arange(trials), 0, a + 1] = 1
            X_new[:, 0, -1] = r
            X = torch.torch.cat((X, X_new), 1)
    return online_regret

config = {
    "lr": tune.loguniform(1e-6, 1e-1),
    "weight_decay": tune.loguniform(1e-3, 1),
    "batch_size": tune.choice([int(2 ** (i / 2)) for i in range(12, 16)]),
}
scheduler = ASHAScheduler(
    metric="best_online_regret",
    mode="min",
    max_t=256,
    reduction_factor=2,
)
result = tune.run(
    train_bandit,
    resources_per_trial={"cpu": 9, "gpu": 1},
    config=config,
    num_samples=25,
    scheduler=scheduler,
)

best_trial = result.get_best_trial("best_online_regret", "min", "all")
best_checkpoint = result.get_best_checkpoint(best_trial, "online_regret", "min")
print(f"Best trial config: {best_trial.config}")
print(f"Best checkpoint online regret: {best_trial.last_result['best_online_regret']}")
print(f"Best checkpoint path: {best_checkpoint.path}")

best_model = TransformerModel(n_states=5)
best_model = nn.DataParallel(best_model)
best_model.to(device)

with best_checkpoint.as_directory() as checkpoint_dir:
    checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
    best_model.load_state_dict(checkpoint_dict["model_state_dict"])

test_online_regret = online_regret(best_model)
print("Best test online regret: {}".format(test_online_regret))