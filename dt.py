import random
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
from scipy.special import softmax

device = torch.device('cpu')

A = 5 # number of actions
N = 80 # number of offline samples
n = 500
N_0 = N * n
gamma = 1 - 1 / n
eps = 1e-10

def generate_B(A=5, N=80, n=500):
    observations, actions, rewards, observations_goal = [], [], [], []
    for _ in range(N):
        p_1 = np.random.dirichlet(np.ones(A), n)
        p_2 = np.zeros((n, A))
        idx = np.random.choice(np.arange(A), n)
        p_2[np.arange(n), idx] = 1
        w = np.random.rand(n, 1)
        p = (1 - w) * p_1 + w * p_2

        cum = p.cumsum(1)
        u = np.random.rand(n, 1)
        a = (u < cum).argmax(1)

        s = np.empty(n, np.int8)
        s[0] = np.random.choice(A)
        s[1:] = a[:-1]
        mu = np.random.rand(A)
        r = np.random.normal(mu[s], 0.3 ** 2)

        s_one_hot = np.zeros((n, A))
        s_one_hot[np.arange(n), s] = 1
        
        a_one_hot = np.zeros((n, A))
        a_one_hot[np.arange(n), a] = 1
        
        observations.append(s_one_hot.tolist())
        actions.append(a_one_hot.tolist())
        rewards.append(r)

        B = np.zeros((n, A + (A + 1) * (n + 1)))
        B[:, :A] = s_one_hot
        B[:, A] = r
        B[:, A + 1:2 * A + 1] = a_one_hot

        for j in range(n):
            B[j, 2 * A + 1:A + (j + 1) * (A + 1)] = B[:j, :A + 1].flatten()
        s_g = np.delete(B, A, 1)
        observations_goal.append(s_g)

    ds = Dataset.from_dict({"observations": observations,
                            "actions": actions,
                            "rewards": rewards,
                            "observations_goal": observations_goal})
    return ds

@dataclass
class DecisionTransformerBanditDataCollator:
    return_tensors: str = "pt"
    max_len: int = 500 # subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 5  # size of action space
    max_ep_len: int = 500 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask, s_g = [], [], [], [], [], [], [], []
        
        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))
            s_g.append(np.array(feature["observations_goal"][si : si + self.max_len]).reshape(1, -1, 2 * self.state_dim + self.max_ep_len * (self.state_dim + 1)))
            
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            s_g[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 2 * self.state_dim + self.max_ep_len * (self.state_dim + 1))), s_g[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        s_g = torch.from_numpy(np.concatenate(s_g, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
            "state_goal":s_g
        }

class NeuralNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(A + (A + 1) * n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

    def forward(self, x):
        v = self.value(x)
        return v

v_model = NeuralNetwork(n).to(device)
v_model.load_state_dict(torch.load('/Users/apple/Downloads/model.pt', map_location=torch.device('cpu'))['model_state_dict'])
v_model.eval()

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, state_goal, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        logits_preds = output[1]
        action_targets = kwargs["actions"]
        rewards = kwargs["rewards"]
        attention_mask = kwargs["attention_mask"]
        act_dim = logits_preds.shape[2]
        
        logits_preds = logits_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        actions_preds = torch.softmax(logits_preds, 1)
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1)[attention_mask.reshape(-1) > 0]
        state_goal = state_goal.reshape(-1, 2 * act_dim + (act_dim + 1) * n)[attention_mask.reshape(-1) > 0] #
        
        with torch.no_grad():
            s_g = torch.cat([state_goal[:, :act_dim], state_goal[:, 2 * act_dim:]], dim=1)
            a_g = state_goal[:, act_dim:]
            v = v_model(s_g)[:, 0]
            tv = v_model(a_g)[:, 0]
        loss = torch.mean(-(rewards + gamma * tv - v + 1) * torch.log(actions_preds[torch.arange(actions_preds.shape[0]), torch.argmax(action_targets, 1)] + eps))
        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

ds_train = generate_B()
ds_eval = generate_B(N=20)
train_collator = DecisionTransformerBanditDataCollator(ds_train)
config = DecisionTransformerConfig(state_dim=train_collator.state_dim, act_dim=train_collator.act_dim, n_positions=2048, n_layer=4, n_head=4)
model = TrainableDT(config)

training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=50,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    warmup_ratio=0.1,
    logging_strategy='epoch',
    optim="adamw_torch",
    max_grad_norm=0.25,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=train_collator,
)

#trainer.train()

def generate_B1(A=5, N=80, n=500):
    observations, actions, rewards, observations_goal = [], [], [], []
    for _ in range(N):
        p_1 = np.random.dirichlet(np.ones(A), n)
        p_2 = np.zeros((n, A))
        idx = np.random.choice(np.arange(A), n)
        p_2[np.arange(n), idx] = 1
        w = np.random.rand(n, 1)
        p = (1 - w) * p_1 + w * p_2

        cum = p.cumsum(1)
        u = np.random.rand(n, 1)
        a = (u < cum).argmax(1)

        s = np.empty(n, np.int8)
        s[0] = np.random.choice(A)
        s[1:] = a[:-1]

        mu = np.random.rand(A)
        r = np.random.normal(mu[s], 0.3 ** 2)

        s_one_hot = np.zeros((n, A))
        s_one_hot[np.arange(n), s] = 1
        
        a_one_hot = np.zeros((n, A))
        a_one_hot[np.arange(n), a] = 1
        
        observations.append(s_one_hot.tolist())
        actions.append(a_one_hot.tolist())
        rewards.append(r)

        B = np.zeros((n, A + (A + 1) * (n + 1)))
        B[:, :A] = s_one_hot
        B[:, A] = r
        B[:, A + 1:2 * A + 1] = a_one_hot

        for j in range(n):
            B[j, 2 * A + 1:A + (j + 1) * (A + 1)] = B[:j, :A + 1].flatten()
        s_g = np.delete(B, A, 1)
        observations_goal.append(s_g)

    ds = Dataset.from_dict({"observations": observations,
                            "actions": actions,
                            "rewards": rewards,
                            "observations_goal": observations_goal})
    return ds, mu

def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)
    #state_goal = state_goal.reshape(1, -1, 2 * model.config.state_dim + model.config.max_ep_len * (model.config.state_dim + 1))

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    #state_goal = state_goal[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
    #state_goal = torch.cat([torch.zeros((1, padding, 2 * model.config.state_dim + model.config.max_ep_len * (model.config.state_dim + 1))), state_goal], dim=1).float()

    state_preds, action_preds, return_preds = model.original_forward( # original_forward
        #state_goal,
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

scale = 1000.0  # normalization for rewards/returns
TARGET_RETURN = 12000 / scale

data = generate_B1(N=1)
ds_test, mu = data
mu_opt = np.max(mu)

episode_return, episode_length = 0, 0
state_dim = 5
act_dim = 5
# device = 'cpu'
# max_ep_len = model.config.max_ep_len
# data = generate_B(N=1)
# ds_test, mu = data

collator = DecisionTransformerBanditDataCollator(ds_test)
state_mean = collator.state_mean.astype(np.float32)
state_std = collator.state_std.astype(np.float32)
state_mean = torch.from_numpy(state_mean).to(device='cpu')
state_std = torch.from_numpy(state_std).to(device='cpu')

state = np.array(ds_test['observations'][0][0])
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(0, device=device, dtype=torch.float32)
#state_goal = torch.zeros((0, 2 * model.config.state_dim + model.config.max_ep_len * (model.config.state_dim + 1)), device=device, dtype=torch.float32)

subopt = np.array([])
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
for t in range(500):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])
    #state_goal = torch.cat([state_goal, torch.zeros((1, 2 * model.config.state_dim + model.config.max_ep_len * (model.config.state_dim + 1)), device=device)], dim=0)

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps
        #state_goal
    )
    actions[-1] = action
    action = action.detach().cpu().numpy()
    action = softmax(action)
    a = np.argmax(action)
    #a = np.random.choice(act_dim, p=action)

    state = np.zeros(state_dim)
    state[a] = 1
    # state = action
    reward = np.random.normal(mu[a], 0.3 ** 2)

    subopt = np.append(subopt, mu_opt - mu[a])

    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1

import matplotlib.pyplot as plt

plt.plot(subopt)
plt.show()