import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


############################################################
# Problem 1a: Volcano Crossing


class VolcanoCrossing():
    """
    grid_world: a 2D numpy array where 0 is explorable, negative integer is a volcano, and positive integer is the goal.
    discount: discount factor
    moveReward: reward of moving from one cell to another
    value_table: a 2D numpy array where each cell represents the value of the cell
    actions: a list of possible actions
    """
    def __init__(self, grid_world, discount=1, moveReward=-1):
        self.grid_world = grid_world
        self.discount = discount
        self.moveReward = moveReward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Return the value table after running |numIters| of value iteration.
    # You do not need to modify this function.
    def value_iteration(self, numIters=1):
        self.value_table = np.zeros(self.grid_world.shape) # Initialize value table

        for _ in range(numIters):
            self.value_table = self.value_update(self.value_table)
        return self.value_table

    # Return the state is Volcano or Island.
    # You do not need to modify this function.
    # If the state is Volcano or Island, return True.
    # Otherwise(self.grid_world[state] == 0), return False.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def is_volcano_or_island(self, state):
        return self.grid_world[state] != 0

    # Checks if the agent can move to the next state.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def movable(self, state, action):
        x, y = state
        i, j = action
        return 0 <= x + i < self.grid_world.shape[0] and 0 <= y + j < self.grid_world.shape[1]


    # Return the value table after updating the value of each grid cell.
    def value_update(self, value_table):
        # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
        new_value_table = np.copy(value_table)
        for x in range(self.grid_world.shape[0]):
            for y in range(self.grid_world.shape[1]):
                if not self.is_volcano_or_island((x, y)):
                    max_value = float('-inf')
                    for action in self.actions:
                        nx, ny = x + action[0], y + action[1]
                        if self.movable((x, y), action):
                            next_value = value_table[nx, ny]
                        else:
                            next_value = value_table[x, y]  # stay in the same place
                        max_value = max(max_value, self.moveReward + self.discount * next_value)
                    new_value_table[x, y] = max_value
                else:
                    # No update needed for terminal states as they are either volcanoes or islands
                    new_value_table[x, y] = self.grid_world[x, y]
        return new_value_table
        # END_YOUR_ANSWER

############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        total, nextCard, counts = state
        if counts is None:
            return []  # No transitions possible from a terminal state

        if action == 'Quit':
            # Terminal state with the sum of cards in hand as the reward.
            return [((total, None, None), 1, total)]

        if action == 'Peek':
            if nextCard is not None:
                # Cannot peek if the next card is already peeked at
                return []
            outcomes = []
            total_cards = sum(counts)
            if total_cards == 0:
                return []  # No cards left to peek at
            for i, count in enumerate(counts):
                if count > 0:
                    # Prob of peeking at card i is the fraction of that card type left
                    prob = count / total_cards
                    new_state = (total, i, counts)
                    outcomes.append((new_state, prob, -self.peekCost))
            return outcomes

        if action == 'Take':
            if nextCard is not None:
                # If nextCard is peeked, draw it deterministically
                card_value = self.cardValues[nextCard]
                new_total = total + card_value
                new_counts = list(counts)
                new_counts[nextCard] -= 1
                if new_total > self.threshold or all(c == 0 for c in new_counts):
                    reward = 0 if new_total > self.threshold else new_total
                    return [((new_total, None, None), 1, reward)]
                else:
                    return [((new_total, None, tuple(new_counts)), 1, 0)]
            else:
                outcomes = []
                total_cards = sum(counts)
                if total_cards == 0:
                    return []  # No cards left to draw
                for i, count in enumerate(counts):
                    if count > 0:
                        card_value = self.cardValues[i]
                        new_total = total + card_value
                        new_counts = list(counts)
                        new_counts[i] -= 1
                        prob = count / total_cards
                        if new_total > self.threshold or all(c == 0 for c in new_counts):
                            reward = 0 if new_total > self.threshold else new_total
                            outcomes.append(((new_total, None, None), prob, reward))
                        else:
                            outcomes.append(((new_total, None, tuple(new_counts)), prob, 0))
                return outcomes

        return []
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        nextQ = max(self.getQ(newState, nextAction) for nextAction in self.actions(newState)) if not isLast(
            newState) else 0
        currentQ = self.getQ(state, action)
        update = self.getStepSize() * (reward + self.discount * nextQ - currentQ)

        for feature, value in self.featureExtractor(state, action):
            self.weights[feature] += update * value
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        nextQ = self.getQ(newState, newAction) if not isLast(newState) else 0
        currentQ = self.getQ(state, action)
        update = self.getStepSize() * (reward + self.discount * nextQ - currentQ)

        for feature, value in self.featureExtractor(state, action):
            self.weights[feature] += update * value
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    deck_counts = counts if counts is not None else []
    standard_value = 1  # uniform feature value for simplicity

    # Constructing the feature list
    features = []

    # Feature based on the total card value linked with the action
    total_feature = ((f"Total_{action}", total), standard_value)
    features.append(total_feature)

    # Binary presence of each card type, indicating availability in the deck
    card_presence = tuple(int(count > 0) for count in deck_counts)
    presence_feature = ((f"Presence_{action}", card_presence), standard_value)
    features.append(presence_feature)

    # Detailed card count features, listed by card types
    for card_type, count in enumerate(deck_counts):
        card_count_feature = ((f"Card_{card_type}_{action}", count), standard_value)
        features.append(card_count_feature)

    return features
    # END_YOUR_ANSWER
