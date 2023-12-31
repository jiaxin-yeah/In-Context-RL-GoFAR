# In-Context-RL-GoFAR
### Step 1
Generate the bandit dataset (`generate_B()`) per the instructions on the Supervised Pretraining paper.
### Step 2
Train the value network and policy network per Algorithm 2 of the GoFAR paper. In our case, we represent goal as the discounted trajectory history, i.e. $g_t=x_t+\gamma\cdot g_{t+1}$ where $x_t$ represents the current trajectory at time $t$: $x_t=(s_t,a_t,r_t)$
### Step 3
Train the Decision-Pretrained Transformer per Algorithm 2 of the Supervised Pretraining paper but sample from the GoFAR policy instead of the optimal policy.
### Step 4
With the trained model $M_\theta(·|s_h, D)$, conduct experiments per Algorithm 3 and 4 of the Supervised Pretraining paper.
