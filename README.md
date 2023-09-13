# In-Context-RL-GoFAR
### Step 1
Generate the bandit dataset (`generate_B()`) per the instructions on the Supervised Pretraining paper.
### Step 2
Train the value network and policy network per Algorithm 2 of the GoFAR paper. In our case, we represent goal as the discounted trajectory history, i.e. $g_t=s_t+a_t+r_t+\gamma^t g_{t-1}$
### Step 3
With the trained policy network model $M_\theta(·|s_h, D)$, conduct experiments per Algorithm 3 and 4 of the Supervised Pretraining paper.
