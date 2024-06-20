# Comparing local search methods

You should use the coordinates `North_America_75.json` and do
multiple runs for each local search. Be sure to experiment with the
default parameter settings to try to get the best results you can.
Then run at least 3 experiments using each search method and compute
the average best cost found.

| HC     | Best Cost |
| ------ | --------- |
| run 1  |     396.8505904527861      |
| run 2  |     382.37057505204626     |
| run 3  |     413.48416172841604     |
| Avg    |   397.5684424110828        |

HC parameters: 
"runs":2,
"steps":200,
"rand_move_prob": 0.25

| SA     | Best Cost |
| ------ | --------- |
| run 1  |    384.53184116646173    |
| run 2  |    371.2097750671252     |
| run 3  |    399.352061911813      |
| Avg    |     385.03122604846664      |

SA parameters:
"runs":25,
"steps":5000,
"init_temp":50,
"temp_decay":0.99

| BS     | Best Cost |
| ------ | --------- |
| run 1  |     305.63349587483754      |
| run 2  |     301.78404254399277      |
| run 3  |     329.30376202859713     |
| Avg    |       312.2404334824758    |

BS parameters:
 "pop_size":50,
"steps":500,
"init_temp":200,
"temp_decay":0.99,
"max_neighbors":10 


Which local search algorithm (HC, SA, or BS) most consistently finds
the best tours and why do you think it outperforms the others?

BS consistently found the best tours. We think this happens because as population based local search, it 
identifies promising areas and focuses on them. It can also simultaneously search many different tours. 


