import abc
from collections import defaultdict
import random
from typing import Dict, List

class ReinforcementLearner(metaclass=abc.ABCMeta):
    """Represents an abstract reinforcement learning agent."""

    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
        """
        self.numStates = numStates
        self.numActions = numActions

        self.epsilon = epsilon
        self.gamma = gamma

        
    @abc.abstractmethod
    def action(self, state: int) -> int:
        """Return learned action for the given state."""
        pass

    @abc.abstractmethod
    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise return learned action for given state."""
        pass

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform the last learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'. Since this is a terminal step, this is a terminal state.
        """
        self.learningStep(step, curState, action, reward, nextState)

    @abc.abstractmethod
    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a learning step of an episode. 

        Args:
            step (int): Index of the current step.
            curState (int): Current state, e.g., s
            action (int): Current action, e.g., a
            reward (float): Observed reward
            nextState (int): Next state, e.g., s'.
        """
        pass


class ModelBasedLearner(ReinforcementLearner):
    """Model-based value iteration reinforcement learning agent."""
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, updateIter: int = 1000, valueConvergence: float = .001, **kwargs):
        super().__init__(numStates, numActions, epsilon, gamma)
        
        self.updateIter = updateIter
        self.valueConvergence = valueConvergence

        # Maintain transition counts and total rewards for each (s, a, s') triple as a list-of-lists-of dictionaries
        # indexed first by state, then by actions. The keys are in the dictionaries are s'.
        self.tCounts: List[List[defaultdict]] = []
        self.rTotal : List[List[defaultdict]]= []
        for _ in range(numStates):
            self.tCounts.append([defaultdict(int) for _ in range(numActions)])
            self.rTotal.append([defaultdict(float) for _ in range(numActions)])

        # Current policy implemented as a dictionary mapping states to actions. Only states with a current policy
        # are in the dictionary. Other states are assumed to have a random policy.
        self.pi: Dict[int, int] = {}

    def action(self, state: int) -> int:
        """Return the action in the current policy for the given state."""
        # Return the specified action in the current policy if it exists, otherwise return
        # a random action
        return self.pi.get(state, random.randint(0, self.numActions - 1))

    def epsilonAction(self, step: int, state: int) -> int:
        """With some probability return a uniform random action. Otherwise return the action in the current policy for the given state."""
        
        
        if random.random() < self.epsilon: # self.epsilon is the probability of taking a random action
            return random.randint(0, self.numActions - 1)
        return self.action(state) # 1 - self.epsilon is the probability of taking the action in the current policy


    def learningStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Perform a value-iteration learning step for the given transition and reward."""
    
        # Update the observed transitions and rewards for (s, a, s') triples. Since we are using
        # defaultdicts we don't need to check if the key exists before incrementing.
        self.tCounts[curState][action][nextState] += 1
        self.rTotal[curState][action][nextState] += reward

        # Update the current policy every updateIter steps
        if step % self.updateIter != 0:
            return
       
        # Implement value iteration to update the policy. 
        # Recall that:
        #   T(s, a, s') = (Counts of the transition (s,a) -> s') / (total transitions from (s,a))
        #   R(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
        # Many states may not have been visited yet, so we need to check if the counts are zero before
        # updating the policy. We will only update the policy for states with state-action pairs that\
        # have been visited.
   
        # Recall value iteration is an iterative algorithm. Here iterate until convergence, i.e., when
        # the change between v_new and v is less than self.valueConvergence for all states.
        v = [0.0] * self.numStates

        while True:
            v_new = v[:]

            for state in range(self.numStates):
                max_value_iteration = float('-inf')
                
                for action in range(self.numActions):
                    total_transitions = sum(self.tCounts[state][action].values())
                    if total_transitions == 0:
                        continue  # Skip if counts are zero

                    value_iteration = 0
                    for next_state, count in self.tCounts[state][action].items():
                        if count == 0:
                            continue
                        transition_probability = count / total_transitions
                        expected_reward = self.rTotal[state][action][next_state] / count
                        value_iteration += transition_probability * (expected_reward + (self.gamma * v[next_state]) )

                    max_value_iteration = max(max_value_iteration, value_iteration)
                
                v_new[state] = max_value_iteration

            # Check for convergence
            if all(abs(new - prev) <= self.valueConvergence for new, prev in zip(v_new, v)):
                break
            v = v_new

        # Update policy based on results of value iteration

        for state in range(self.numStates):
                best_action = None
                max_value_iteration = float('-inf')

                for action in range(self.numActions):
                    total_transitions = sum(self.tCounts[state][action].values())
                    if total_transitions == 0:
                        continue
                    value_iteration = 0

                    for next_state, count in self.tCounts[state][action].items():
                        if count == 0:
                            continue
                        transition_probability = count / total_transitions
                        expected_reward = self.rTotal[state][action][next_state] / count
                        value_iteration += transition_probability * (expected_reward + (self.gamma * v_new[next_state]) )

                    if value_iteration > max_value_iteration:
                        max_value_iteration = value_iteration
                        best_action = action

                self.pi[state] = best_action




class QLearner(ReinforcementLearner):
    """Q-learning-based reinforcement learning agent."""
    
    def __init__(self, numStates: int, numActions: int, epsilon: float, gamma: float, alpha: float = 0.1, initQ: float=0.0, **kwargs):
        """Initialize GridWorld reinforcement learning agent.

        Args:
            numStates (int): Number of states in the MDP.
            numActions (int): Number of actions for each state in the MDP.
            epsilon (float): Probability of taking a random action.
            gamma (float): Discount parameter.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            initQ (float, optional): Initial Q value. Defaults to 0.0.
        """
        super().__init__(numStates, numActions, epsilon=epsilon, gamma=gamma)

        self.alpha = alpha

        # The Q-table, q, is a list-of-lists, indexed first by state, then by actions
        self.q: List[List[float]] = []  
        for _ in range(numStates):
            self.q.append([initQ] * numActions)

    def action(self, state: int) -> int:
        """Returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        # TODO: Implement greedy action selection
        return 0

    def epsilonAction(self, step: int, state: int) -> int:
        """With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly)."""
        # TODO: Implement epsilon-greedy action selection
        return 0

    def learningStep(self, step: int, curState, action, reward, nextState):
        """Performs a Q-learning step based on the given transition, action and reward."""
        # TODO: Implement the Q-learning step

    def terminalStep(self, step: int, curState: int, action: int, reward: float, nextState: int):
        """Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0."""
        # TODO: Implement the terminal step of the learning algorithm
