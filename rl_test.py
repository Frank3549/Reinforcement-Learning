import sys, unittest
import rl



class ModelBasedLearnerTest(unittest.TestCase):
    def test_epsilon_action(self):
        """Epsilon action selection returns random action with probability epsilon"""
        learner = rl.ModelBasedLearner(2, 2, epsilon=0.5, gamma=0.9)
        learner.pi = {0: 1, 1: 0}
        observed_actions = [set(), set()]
        for _ in range(100):
            for state in range(2):
                action = learner.epsilonAction(1, state)
                self.assertIn(action, [0, 1], msg="Action must be 0 or 1")
                observed_actions[state].add(action)
        self.assertEqual(len(observed_actions[0]), 2, msg="Expected both actions in with eps=.5 over 100 trials")   
        self.assertEqual(len(observed_actions[1]), 2, msg="Expected both actions in with eps=.5 over 100 trials")   

    def test_learning_step_nonupdate(self):
        """Model-based learningStep only update policy on select iterations"""
        learner = rl.ModelBasedLearner(2, 2, epsilon=0.1, gamma=0.9, updateIter=10)
        self.assertEqual(len(learner.pi), 0)
        learner.learningStep(1, 0, 1, 2, 1)
        self.assertEqual(len(learner.pi), 0, msg="Policy should only updated on select iterations")

    def test_learning_step_update(self):
        """Model-based learningStep updates policy on desired iteration"""
        learner = rl.ModelBasedLearner(2, 2, epsilon=0.1, gamma=0.9, updateIter=10)
        self.assertEqual(len(learner.pi), 0)
        learner.learningStep(8, 0, 0, 3, 0)
        learner.learningStep(9, 1, 0, 1, 0)
        learner.learningStep(10, 0, 1, 2, 1)
        self.assertEqual(learner.pi, { 0: 0, 1: 0})


class QLearnerTest(unittest.TestCase):
    def test_action(self):
        """Greedy action selection returns action with highest Q-value"""
        learner = rl.QLearner(2, 2, epsilon=0.1, gamma=0.9, alpha=0.1, initQ=0)
        learner.q = [[0, 1], [1, 0]]
        self.assertEqual(learner.action(0), 1)
        self.assertEqual(learner.action(1), 0)

    def test_greedy_tie_breaking(self):
        """Greedy action selection returns action with highest Q-value breaking ties randomly"""
        learner = rl.QLearner(2, 2, epsilon=0.1, gamma=0.9, alpha=0.1, initQ=0)
        learner.q = [[1, 1], [1, 0]]
        observed_actions = set()
        for _ in range(100):
            action = learner.action(0)
            self.assertIn(action, [0, 1])
            observed_actions.add(action)
        self.assertEqual(len(observed_actions), 2, msg="Expected both actions in with ties broken randomly over 100 trials")

    def test_epsilon_greedy(self):
        """Epsilon-greedy action selection returns random action with probability epsilon"""
        learner = rl.QLearner(2, 2, epsilon=0.5, gamma=0.9, alpha=0.1, initQ=0)
        learner.q = [[0, 1], [1, 0]]
        observed_actions = [set(), set()]
        for _ in range(100):
            for state in range(2):
                action = learner.epsilonAction(1, state)
                self.assertIn(action, [0, 1], msg="Action must be 0 or 1")
                observed_actions[state].add(action)
        self.assertEqual(len(observed_actions[0]), 2, msg="Expected both actions in with eps=.5 over 100 trials")   
        self.assertEqual(len(observed_actions[1]), 2, msg="Expected both actions in with eps=.5 over 100 trials")   

    def test_epsilon_greedy_with_zero_epsilon(self):
        """Epsilon-greedy action with epsilon=0 with same as greedy action"""
        learner = rl.QLearner(2, 2, epsilon=0.0, gamma=0.9, alpha=0.1, initQ=0)
        learner.q = [[0, 1], [1, 0]]
        self.assertEqual(learner.action(0), 1)
        self.assertEqual(learner.action(1), 0)

    def test_learning_step(self):
        """Learning step updates Q-value based on reward"""
        learner = rl.QLearner(2, 2, epsilon=0.0, gamma=0.9, alpha=0.1, initQ=0)
        
        learner.q = [[.25, .3], [.4, .1]]
        learner.learningStep(1, 0, 1, 2, 1)
 
        self.assertAlmostEqual(learner.q[0][0], 0.25, msg="Other Q-values should remain unchanged")
        self.assertAlmostEqual(learner.q[1][0], 0.4, msg="Other Q-values should remain unchanged")
        self.assertAlmostEqual(learner.q[1][1], 0.1, msg="Other Q-values should remain unchanged")
        self.assertAlmostEqual(learner.q[0][1], 0.506)

    def test_terminal_step(self):
        """Terminal step updates Q-value based on reward"""
        learner = rl.QLearner(2, 2, epsilon=0.0, gamma=0.9, alpha=0.1, initQ=0)
        learner.q[0] = [0, 1]
        learner.terminalStep(1, 0, 1, 3, 2)
        self.assertAlmostEqual(learner.q[0][1], 1.2)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
