# CS311 Programming Assignment 6: Reinforcement Learning

For this assignment, you will be implementing reinforcement learning algorithms. Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code trains and tests reinforcement learning algorithms on a gridworld. Executing `gridworld.py` will train and test your program on a simple world by default. You can change the grid and the learning parameters by changing the optional arguments shown below.

```
usage: gridworld.py [-h] [-w WORLD] [-l {value,q}] [-a ALPHA] [-e EPSILON] [-g GAMMA] [-i INITQ] [-t TRIALS] [-p EPISODES] [-m MAXSTEPS] [-d N] [-b BIGQ] [-u UPDATE] [-c CONVERGENCE]

Use reinforcement learning algorithms to solve gridworld problems.

options:
  -h, --help            show this help message and exit
  -w WORLD, --world WORLD
                        File with grid layout (default: grid.txt)
  -l {value,q}, --learner {value,q}
                        Learning algorithm to use (default: q)
  -a ALPHA, --alpha ALPHA
                        Step-size alpha (default: 0.1)
  -e EPSILON, --epsilon EPSILON
                        Exploration rate epsilon (default: 0.1)
  -g GAMMA, --gamma GAMMA
                        Discount factor gamma (default: 0.9)
  -i INITQ, --initQ INITQ
                        Initial Q-value (default: 0)
  -t TRIALS, --trials TRIALS
                        Number of trials to run (default: 1)
  -p EPISODES, --episodes EPISODES
                        Number of episodes per trial (default: 500)
  -m MAXSTEPS, --maxsteps MAXSTEPS
                        Maximum number of steps per episode (default: 100)
  -d N, --display N     Display every Nth episode (has no effect if TRIALS > 1) (default: 1)
  -b BIGQ, --bigQ BIGQ  Biggest possible magnitude for Q-values for display purposes (default: the largest reward value) (default: None)
  -u UPDATE, --update UPDATE
                        Update interval for model-based learning (default: 1000)
  -c CONVERGENCE, ---convergence CONVERGENCE
                        Threshold for convergence in value iteration (default: 0.001)
```

For example to run the program with the value-iteration model-based learner for 50000 episodes, printing results every 1000 episodes: `python3 gridworld.py -l value -p 50000 -d 1000`.

If you are working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run gridworld.py -l value -p 50000 -d 1000`.

# Unit testing

To assist you during development, a unit test suite is provided in `rl_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `rl_test.py` file as a program, e.g. `python3 rl_test.py`. 

```
python3 rl_test.py 
.........
----------------------------------------------------------------------
Ran 9 tests in 0.002s

OK
```

## Grid world text-file format

In the grid world text file, the first row has the dimensions of the grid. In the grid itself, each entry has three comma-separated entries. The first is either "." or "#", indicating whether that space is a wall or not. The second is the reward for entering that square. The third is either "T" or "F", indicating whether that square is terminal or not.

## Credits

This assignment was adapted from the WARLACS AI assignments by Erin J. Talvitie as presented in Model AI assignments 2023.

Neller, T. W., Walker, R., Dias, O., Yalçın, Z., Breazeal, C., Taylor, M., Donini, M., Talvitie, E. J., Pilgrim, C., Turrini, P., Maher, J., Boutell, M., Wilson, J., Norouzi, N., & Scott, J. (2024). Model AI Assignments 2023. Proceedings of the AAAI Conference on Artificial Intelligence, 37(13), 16104-16105. https://doi.org/10.1609/aaai.v37i13.26913