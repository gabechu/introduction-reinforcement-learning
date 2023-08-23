# What
Gridworld exploration. Imagine a 5x5 grid world where you're on a quest to explore the rewards following a specific policy. The implementation estimates a random policy and an optimal policy.

# How
We calcualte the state-value functions with iterations. Run `poetry run python -m src.chapter_3.gridworld.run` to execute the program.

# Results Snapshot
## Random Policy:
After 109 steps, the random policy converges, resulting in the state-value matrix:
```
[[ 3.30899634  8.78929186  4.42761918  5.3223676   1.49217876]
 [ 1.52158807  2.99231786  2.25013995  1.90757171  0.54740271]
 [ 0.05082249  0.73817059  0.67311326  0.35818622 -0.40314114]
 [-0.9735923  -0.43549543 -0.35488227 -0.58560509 -1.18307508]
 [-1.85770055 -1.34523126 -1.22926726 -1.42291815 -1.97517905]]
```
## Optimal Policy:
After 47 steps, the optimal policy converges, yielding the state-value matrix:
```
[[21.97748529 24.4194281  21.97748529 19.4194281  17.47748529]
 [19.77973676 21.97748529 19.77973676 17.80176308 16.02158677]
 [17.80176308 19.77973676 17.80176308 16.02158677 14.4194281 ]
 [16.02158677 17.80176308 16.02158677 14.4194281  12.97748529]
 [14.4194281  16.02158677 14.4194281  12.97748529 11.67973676]]
 ```