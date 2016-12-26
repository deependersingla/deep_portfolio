# rl_portfolio
This Repository uses Reinforcement Learning to Optimize portfolio allocation. The goal is to make profitable agent which can optimize portfolio between n stocks (or any assets) by taking there price series data and output holdings in each. Portfolio can be long-short. 

#How it works:
In the start, I will be using episode length of one day and price series data for last 2,3 days for each stock.Time stamp will be per minute for both input and output in the start. I will start with 5 stocks. Mostly stock from same sector to start with so algo can also find some covariance among them. Idea in the end is to trade one index all stocks. 

#More Detailed Explanation of Algorithm:
a) Make LSTM network for each asset and predict next interval price. All these networks are trained individually and trained network parameter is stored. (Code inside supervised learning) <br>  
b) Use trained LSTM networks and with other layer combine the network and start training network using Reinforcement learning so second network predict allocation to each. 

The above idea is basically first LSTM networks are learning individual stock movements. In second network they combine each of this for better portfolio optimization using covariance among them etc.

