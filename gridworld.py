import argparse
from rl import *

class GridWorld:
    """Represents a grid world problem."""

    def __init__(self, filename):
        """Takes the name of a file that contains the layout of the grid. The first line of the file should contains the dimensions of the grid: number of rows and number of columns, in that order. Then each entry in the grid should have three comma-separated elements:
        - One of *, ., or # to represent the start state, a blank space, or a wall, respectively
        - The reward for entering that square
        - Either T or F to indicate whether the state is terminal"""

        self.__grid = []
        self.__curState = None

        self.__maxR = -float("inf")
        self.__minR = float("inf")

        with open(filename, "r") as fin:
            self.__dims = [int(d) for d in fin.readline().split()]

            for i in range(self.__dims[0]):
                self.__grid.append([])
                tokens = fin.readline().split()

                if len(tokens) != self.__dims[1]:
                    raise ValueError(
                        f"Length of row {i} does not match given width: {len(tokens)} (expected {self.__dims[1]})"
                    )

                for j in range(len(tokens)):
                    attributes = tokens[j].split(",")  # (!)Wall, reward, (!)terminal

                    if len(attributes) != 3:
                        raise ValueError(
                            f"Unexpected number of attributes for position {(i, j)}: {len(attributes)} expected 3 comma-separated values)"
                        )

                    if attributes[0] not in [".", "#", "*"]:
                        raise ValueError(
                            f"Unexpected layout symbol for position {(i, j)}: {attributes[0]} (expected ., #, or *)"
                        )

                    if attributes[0] == "*":  # Starting location
                        if self.__curState != None:
                            raise ValueError(
                                f"Multiple start states detected at {self.__curState} and {(i, j)}"
                            )

                        self.__curState = (i, j)
                        attributes[0] = "."

                    attributes[1] = float(attributes[1])
                    self.__maxR = max(attributes[1], self.__maxR)
                    self.__minR = min(attributes[1], self.__minR)

                    if attributes[2] == "T":
                        attributes[2] = True
                    elif attributes[2] == "F":
                        attributes[2] = False
                    else:
                        raise ValueError(
                            f"Unexpected terminal indicator at position {(i, j)}: {attributes[2]} (expected T or F)"
                        )

                    self.__grid[-1].append(attributes)

        if self.__curState == None:
            raise ValueError("No start state detected.")
        else:
            self.__initState = self.__curState

        self.__actionMap = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def reset(self):
        """Resets the problem to the initial state."""
        self.__curState = self.__initState

    def transition(self, action):
        """Moves the agent in the grid.

        The actions represent the cardinal directions. Actions 0, 1, 2, and 3 represent
        North, East, South, and West, respectively. Trying to move onto a wall or off the
        grid results in no movement.

        Returns the reward from the transition.
        """
        if self.isTerminal():
            return 0

        if action not in range(4):
            raise ValueError("Invalid action: " + str(action))

        direction = self.__actionMap[action]

        newLoc = (self.__curState[0] + direction[0], self.__curState[1] + direction[1])

        if self.isInBounds(newLoc) and not self.isWall(newLoc):
            self.__curState = newLoc

        return self.getReward(self.__curState)

    def getNumStates(self):
        return self.__dims[0] * self.__dims[1]

    def getState(self):
        """Gets the current state as a number between 0 and width*height-1."""
        return self.__curState[0] * self.__dims[1] + self.__curState[1]

    def getAgentLoc(self):
        """Returns the current state as a tuple: (row, column)."""
        return self.__curState[0], self.__curState[1]

    def __str__(self):
        stateStr = ""
        for i in range(len(self.__grid)):
            for j in range(len(self.__grid[i])):
                if (i, j) == self.__curState:
                    stateStr += "*"
                else:
                    stateStr += self.__grid[i][j][0]
            stateStr += "\n"
        return stateStr

    def isInBounds(self, loc):
        """Determines if a given location (row, column) is inside the grid."""
        return (
            loc[0] >= 0
            and loc[0] < self.__dims[0]
            and loc[1] >= 0
            and loc[1] < self.__dims[1]
        )

    def isTerminal(self):
        """Returns True if the environment is currently in a terminal state."""
        return self.isTerminalLoc(self.__curState)

    def isTerminalLoc(self, loc):
        """Determines if a given location (row, column) is terminal."""
        return self.__grid[loc[0]][loc[1]][2]

    def isWall(self, loc):
        """Determines if a given location (row, column) is a wall."""
        return self.__grid[loc[0]][loc[1]][0] == "#"

    def getReward(self, loc):
        """Returns the reward associated with the given location (row, column)."""
        return self.__grid[loc[0]][loc[1]][1]

    def getDims(self):
        """Returns a tuple containing the number of rows and number of columns in the grid."""
        return tuple(self.__dims)

    def getMaxReward(self):
        """Returns the maximum reward value in the grid."""
        return self.__maxR

    def getMinReward(self):
        """Returns the minimum reward value in the grid."""
        return self.__minR


def main():
    parser = argparse.ArgumentParser(
        description="Use reinforcement learning algorithms to solve gridworld problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-w",
        "--world",
        type=str,
        default="grid.txt",
        help="File with grid layout",
    )
    parser.add_argument(
        "-l",
        "--learner",
        type=str,
        choices=["value", "q"],
        default="q",
        help="Learning algorithm to use",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.1,
        help="Step-size alpha",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate epsilon",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor gamma",
    )
    parser.add_argument("-i", "--initQ", type=float, default=0, help="Initial Q-value")
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of trials to run"
    )
    parser.add_argument(
        "-p",
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes per trial",
    )
    parser.add_argument(
        "-m",
        "--maxsteps",
        type=int,
        default=100,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "-d",
        "--display",
        metavar="N",
        type=int,
        default=1,
        help="Display every Nth episode (has no effect if TRIALS > 1)",
    )
    parser.add_argument(
        "-b",
        "--bigQ",
        type=float,
        help="Biggest possible magnitude for Q-values for display purposes (default: the largest reward value)",
    )
    parser.add_argument(
        "-u",
        "--update",
        type=int,
        default=1000,
        help="Update interval for model-based learning",
    )
    parser.add_argument(
        "-c",
        "---convergence",
        type=float,
        default=0.001,
        help="Threshold for convergence in value iteration",
    )

    args = parser.parse_args()

    world = GridWorld(args.world)

    avgTotal = [0] * args.episodes
    avgDiscounted = [0] * args.episodes
    avgSteps = [0] * args.episodes
    avgFinalTotal = 0
    avgFinalDiscounted = 0
    avgFinalSteps = 0

    for trial in range(args.trials):
        if args.trials > 1:
            print("Trial", trial + 1, end=" ")

        if args.learner == "value":
            agent = ModelBasedLearner(
                world.getNumStates(),
                4,
                epsilon=args.epsilon,
                gamma=args.gamma,
                updateIter=args.update,
                valueConvergence=args.convergence,
            )
        elif args.learner == "q":
            agent = QLearner(
                world.getNumStates(),
                4,
                epsilon=args.epsilon,
                gamma=args.gamma,
                alpha=args.alpha,
                initQ=args.initQ,
            )

        total_step = 0
        for ep in range(args.episodes):
            totalR = 0.0
            discountedR = 0.0
            discount = 1.0

            world.reset()

            curState = world.getState()
            action = agent.epsilonAction(total_step, curState)
            reward = world.transition(action)
            totalR += reward
            discountedR += discount * reward
            total_step += 1

            ep_step = 1
            while not world.isTerminal() and ep_step < args.maxsteps:
                agent.learningStep(
                    total_step, curState, action, reward, world.getState()
                )

                curState = world.getState()
                action = agent.epsilonAction(total_step, curState)
                reward = world.transition(action)
                totalR += reward
                discount *= args.gamma
                discountedR += discount * reward

                ep_step += 1
                total_step += 1

            if world.isTerminal():
                agent.terminalStep(
                    total_step, curState, action, reward, world.getState()
                )
            else:
                agent.learningStep(
                    total_step, curState, action, reward, world.getState()
                )

            if args.trials == 1 and (ep == 0 or (ep + 1) % args.display == 0):
                print(f"Episode {ep + 1}:", totalR, discountedR, ep_step)

            avgTotal[ep] += totalR
            avgDiscounted[ep] += discountedR
            avgSteps[ep] += ep_step

        world.reset()

        totalR = 0
        discountedR = 0
        discount = 1
        curState = world.getState()
        action = agent.action(curState)
        reward = world.transition(action)
        totalR += reward
        discountedR += discount * reward
        step = 1
        while not world.isTerminal() and step < args.maxsteps:
            curState = world.getState()
            action = agent.action(curState)
            reward = world.transition(action)
            totalR += reward
            discount *= args.gamma
            discountedR += discount * reward

            step += 1

        print("Final policy:", totalR, discountedR, step, flush=True)

        avgFinalTotal += totalR
        avgFinalDiscounted += discountedR
        avgFinalSteps += step

    for i in range(args.episodes):
        if args.trials > 1 and (i == 0 or (i + 1) % args.display == 0):
            print(
                f"Average episode {i+1}:",
                avgTotal[i] / args.trials,
                avgDiscounted[i] / args.trials,
                avgSteps[i] / args.trials,
            )

    if args.trials > 1:
        print(
            "Average final greedy policy:",
            avgFinalTotal / args.trials,
            avgFinalDiscounted / args.trials,
            avgFinalSteps / args.trials,
        )


if __name__ == "__main__":
    main()
