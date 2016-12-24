# rl_portfolio
This Repository uses Reinforcement Learning to Optimize portfolio allocation. The goal is to make profitable agent which can optimize portfolio between n stocks (or any assets) by taking there price series data and output holdings in each. Portfolio can be long-short. 

#How it works:
In the start, I will be using episode lenght of one day and price series data for last 2,3 days for each stock.Time stamp will be per minute for both input and output in the start. I will start with 5 stocks. Mostly stock from same sector to start with so algo can also find some covariance among them. Idea in the end is to trade one index all stocks. 