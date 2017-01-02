# rl_portfolio
This Repository uses Reinforcement Learning and Supervised learning to Optimize portfolio allocation. The goal is to make profitable agent which can optimize portfolio between n stocks (or any assets) by taking there price series data and output holdings in each. Portfolio can be long-short. 

#How it works:
In the start, I will be using episode length of one day and price series data for last n days for each stock. Time stamp will be per minute for both input and output.Mostly stock will be from same sector to start with so algo can also find some covariance among them. Idea in the end is to trade different assets and make a balanced online optimized portfolio.

#More Detailed Explanation of Algorithm:
a) Make LSTM network for each asset and predict next interval price. All these networks are trained individually and trained network parameter is stored. (Code inside supervised learning) <br>  
b) Use trained LSTM networks and with other layer combine the network and start training network using Reinforcement learning so second network predict allocation to each. 
