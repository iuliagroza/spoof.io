# spoof.io - A Proximal Policy Optimization Approach to Detect Spoofing in Algorithmic Trading
📜 [PAPER](https://github.com/iuliagroza/spoof.io/blob/main/thesis/IEEE_A_Proximal_Policy_Optimization_Approach_to_Detect_Spoofing_in_Algorithmic_Trading.pdf) <br>
📕 [THESIS](https://github.com/iuliagroza/spoof.io/blob/main/thesis/A_Proximal_Policy_Optimization_Approach_to_Detect_Spoofing_in_Algorithmic_Trading.pdf)

**spoof.io** is a proof-of-concept visualisation tool that simulates historical Level 3 LOB data from the **LUNA flash crash from May 2022**, and outputs in real time orders detected as spoofing attempts. <br><br> Among the most pressing issues in algorithmic trading are spoofing tactics, where traders deceive other market participants by placing and then quickly canceling large orders. This leads to artificial price movements and compromises the trust in market fairness. <br><br> This project explores the development of a **Proximal Policy Optimization (PPO)** agent to detect spoofing in real time using **Level 3 Limit Order Book** data from the LUNA flash crash in May 2022. We develop an anomaly detection system to label legitimate spoofing attempts. Ultimately, we simulate the market environment serving as the playground for our policy network. <br><br> Our core contributions include the integration of Proximal Policy Optimization in market surveillance, the use of Level 3 Limit Order Book data in a machine learning solution to harness temporal data, and the feature engineering of rolling statistics for price and size movements. Our experimental results demonstrate the feasibility of deep reinforcement learning in advancing market integrity: **training loss** converged to **0.13** and obtained a **total reward of 9,500**, with minimal volatility.

## Run locally
1. Open Redis server
   ```bash
   $ cd webapp/backend
   $ redis-server
   ```
2. Run simulation server
   ```bash
   $ cd webapp/backend
   $ daphne -p 8000 backend.asgi:application
   ```
3. Run web client
   ```bash
   $ cd webapp/frontend
   $ npm start
   ```
