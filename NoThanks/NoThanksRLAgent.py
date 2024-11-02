from NoThanksPlayer import Player
from random import randint
import random 
from collections import defaultdict
import copy
import pickle

global_epsilon = 0
start_from_scratch = False
global_training = False
trained_table = 'trained_qtable_10_mil_new_reward.pkl'
global_testing = True

class RLPlayer(Player):
    """
    This class is a player that uses reinforcement learning to play the game of No Thanks!
    The agent uses tabular Q-learning to learn the optimal policy with epsilon-greedy exploration.
    The state is represented as a string, and the Qtable is a dictionary with the state as the key 
    and the value is a list of the Q-values for taking and passing the card.

    The Q-table is retained between games, so the agent can learn from previous games.
    """
    # Initialize the Qtable and the epsilon value based on the training/testing status
    if start_from_scratch and global_training:
        Qtable = defaultdict(float)
        global_epsilon = 0.9
    #load the trained Qtable
    elif (not start_from_scratch) and global_training:
        with open(trained_table, 'rb') as f:
            Qtable = pickle.load(f)
        global_epsilon = 0.9
    elif global_testing:
        with open(trained_table, 'rb') as f:
            Qtable = pickle.load(f)
        global_epsilon = 0
    
    episode_count = 1
    takes = 0
    totalActions = 0
    def __init__(self, name=""):
        super().__init__(name)
        self.setName("RL Player 1")
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = RLPlayer.global_epsilon
        self.training = global_training
    

    def stateToString(self, card):
        """
        Converts the state to a string representation
        Simplified state: <player's highest card, player's lowest card, player's coins, card on offer>
        """
        high_card = max(self.collection, key=lambda x: x.number) if self.collection else 0
        low_card = min(self.collection, key=lambda x: x.number) if self.collection else 0
        stateStr = f"{high_card},{low_card},{self.coins};{card.number};{card.coins}"
        return stateStr
    
    def takeSmallAction(self, card,state):
        """
        This method is called when the player is exploring.
        Instead of a completely random action, which would mean exploring the entire state space,
        the player uses the best default AI strategy to take a card if the penalty is less than 10.
        """
        if self.coins <= 0:
            return True
        if card.penalty <= 10:
            return True
        return False

    def take(self, card, state):
        """
        The default take method is called when the player needs to take an action.
        The player uses epsilon-greedy exploration to take an action.
        Updates a global counter for the number of actions taken and the number of times the player takes a card.

        args: 
        card - the card on offer
        state - the state of the game

        returns:
        action - True if the player takes the card, False if the player passes
        """
        RLPlayer.episode_count += 1
        action = self.epsilonGreedy(card, state)
        if action:
            RLPlayer.takes += 1
        RLPlayer.totalActions += 1
        return(action)
            
    
    def epsilonGreedy(self, card, state):
        """
        This method implements epsilon-greedy exploration.
        The player takes a random action with probability epsilon, and takes the best action with probability 1-epsilon.
        The player uses the best default AI strategy to take a card if exploring or if the Q-values for taking and passing are equal.
        
        args:
        card - the card on offer
        state - the state of the game

        returns:
        action - True if the player takes the card, False if the player passes
        """
        self.currentState = self.stateToString(card)
        
        # Explore with probability epsilon
        if random.uniform(0,1) < self.epsilon:
            action = self.takeSmallAction(card,state)
            self.reward = self.calculateReward(card, action)
            newStateStr = self.getNewState(card, action, state)
            self.currentStateQ = RLPlayer.Qtable[self.currentState, action]
            # update the Qtable with the Q-learning formula if training
            if self.training:
                RLPlayer.Qtable[self.currentState, action] = (1- self.alpha)*self.currentStateQ + self.alpha * ((self.reward + self.gamma * self.getBestQ(newStateStr)) - self.currentStateQ)
            return action
        else:
            # Exploit with probability 1-epsilon
            q_take = RLPlayer.Qtable[self.currentState, True]
            q_pass = RLPlayer.Qtable[self.currentState, False]
            if q_take > q_pass:
                action = True
                self.reward = self.calculateReward(card, action)
                newStateStr = self.getNewState(card, action, state)
                self.currentStateQ = RLPlayer.Qtable[self.currentState, action]
                # update the Qtable with the Q-learning formula if training
                if self.training:
                    RLPlayer.Qtable[self.currentState, action] = (1 - self.alpha) * self.currentStateQ + self.alpha * ((self.reward + self.gamma * self.getBestQ(newStateStr)) - self.currentStateQ)
                return action
            if q_take == q_pass:
                action = self.takeSmallAction(card,state)
                self.reward = self.calculateReward(card, action)
                newStateStr = self.getNewState(card, action, state)
                self.currentStateQ = RLPlayer.Qtable[self.currentState, action]
                if self.training:
                    RLPlayer.Qtable[self.currentState, action] = (1-self.alpha) * self.currentStateQ + self.alpha * ((self.reward + self.gamma * self.getBestQ(newStateStr)) - self.currentStateQ)
                return action          
            else:
                action = False
                self.reward = self.calculateReward(card, action)
                newStateStr = self.getNewState(card, action, state)
                self.currentStateQ = RLPlayer.Qtable[self.currentState, action]
                # update the Qtable with the Q-learning formula if training
                if self.training:
                    RLPlayer.Qtable[self.currentState, action] = (1-self.alpha) * self.currentStateQ + self.alpha * ((self.reward + self.gamma * self.getBestQ(newStateStr)) - self.currentStateQ)
                return action

    def calculateReward(self, card, action):
        """
        This method calculates the reward for taking or passing a card.

        args:
        card - the card on offer
        action - True if the player takes the card, False if the player passes

        returns:
        reward - the reward for taking or passing the card
        """
        current_penalty = self.penalty()
        average_penalty = self.totalpenalty
        future_penalty = self.penaltyWhenTake(card)
        penalty_difference = future_penalty - current_penalty
        if self.games > 0:
            average_penalty = self.totalpenalty / self.games
        if action :
            reward = card.coins  # Start with the coins on the card
            if penalty_difference > 0:
                # Disincentive to take if the penalty increased by taking the card
                reward -= (abs(penalty_difference) + average_penalty + 50)
            else:  
                # Incentive to take if the penalty decreased (or stayed the same)
                reward  -= ((abs(penalty_difference) + average_penalty) - 100) 
        else:  # Passing the card strategically 
            if penalty_difference <= 0:
                # Disincentive for passing on a card that would decrease the penalty or keep it the same
                reward = -((abs(penalty_difference) + average_penalty) + 50)
            else:  
                # Incentive for passing on a card that would have increased the penalty
                reward = -((abs(penalty_difference) + average_penalty) - 50)
        return reward

    
    def getBestQ(self, state):
        """
        This method returns the best Q-value for a given state.

        args:
        state - the state of the game

        returns:
        bestQ - the best Q-value for the state
        """
        q_take = RLPlayer.Qtable[state, True]
        q_pass = RLPlayer.Qtable[state, False]
        if q_take > q_pass:
            return q_take
        if q_take == q_pass:
            return random.choice([q_take, q_pass])
        else:
            return q_pass
        
    def getNewState(self, card, action, state):
        """
        This method returns the new state after taking an action.
        Makes a copy of self (to have access to the player methods) and returns the new state after taking the action
        Copy is needed to avoid changing the state of the game outside of the player class
        """
        newSelf = copy.deepcopy(self)
        if action:
            newSelf.addCardToCollection(card)
            newStateStr = newSelf.stateToString(card)
            return newStateStr
        else:
            newSelf.coins -= 1
            newStateStr = newSelf.stateToString(card)
            return  newStateStr
    