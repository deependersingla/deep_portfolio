# rl_portfolio
This Repository uses Reinforcement Learning and Supervised learning to Optimize portfolio allocation. The goal is to make profitable agent which can optimize portfolio between n stocks (or any assets) by taking there price series data and output holdings in each. Portfolio can be long-short. 

#How it works:
In the start, I will be using episode length of one day and price series data for last n days for each stock. Time stamp will be per minute for both input and output.Mostly stock will be from same sector to start with so algo can also find some covariance among them. Idea in the end is to trade different assets and make a balanced online optimized portfolio.

#More Detailed Explanation of Algorithm:
a) Make LSTM network for each asset and predict next interval price. All these networks are trained individually and trained network parameter is stored. (Code inside supervised learning) <br>  
b) Use trained LSTM networks and with other layer combine the network and start training network using Reinforcement learning so second network predict allocation to each. 

#Other thoughts on the project are:
a) Most of the financial engineering is about predicting the next price interval of the underlying asset. If you see first part that's what algorithm do it takes a supervised learning (made using LSTM) to predict that. This part can be separately optimised and made better in future. The more better it become the more better the algo will able to make portfolio. 
<br>
b) Second part making a portfolio using RL: This part takes input from first part and log returns of the price series to predict portfolio allocation. Idea here is that even knowing next price interval doesn't guarantee much, unless you can make a portfolio of different assets which can have less drawdown and consistent returns. <br>

Here, Idea is to make automatic optimisation algorithm based on how we set rewards. For example:<br>

1) Make a portfolio where I only care about returns: Just set the reward based on returns, Network will output algorithm which only trying to optimise returns.<br>
  
2) Make a portfolio where drawdown should be less and medium returns, in this case rewards will be functions which will use drawdown also to give feedback to the network. The portfolio made will be focussing on drawdown also.  The brief idea is to make algorithm which can be improved everyday and so much general purpose that it can do based on what we want.<br>

#Finding which network can explore:
a) Automatic finding of look_back period for momentum strategy.<br>
b) Automatic finding of average period of mean reverting strategy.<br>
c) Optimize portfolio to autocorrect above.<br>
