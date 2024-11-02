from Take5Player import Player
from Take5State import State, PlayerState
from Take5Game import Card, Row
from Take5 import Game
from Take5 import Play
from collections import defaultdict
import random
import copy

class RLAgent(Player):
	"""
	Implements a reinforcement learning agent that learns to play the game of Take5 using
	Linear function approximation with Q-learning

	NOTE : 
	There was major refactoring done to the Take5.py file to make it work with the RLAgent during training
	for example, the simulate round method requires an object which deviates from the original implementation 
	of the state representation.
	"""
	def __init__(self):
		"""
		Initialize the RLAgent
		Initialize weights with random values from a normal distribution mean = 0, std = 1
		The features are :	1. value of the card,
	 					 	2. penalty of the card,
						 	3. difference between the card and the highest card on the table that is lower than the card being considered, 
						 	4. number of cards played in the nearest row,
						 	5. penalty in the nearest row
						 	6. total accumulated penalty
		"""
		Player.__init__( self )
		self.setName('RLAgent')
		self.epsilon = 0
		self.weights_play_card = [0.7220845337755154, -0.9287026064802018, -0.6470870910600508, 0.6987392801379517, -1.1764169474551047, -0.4199210186404511]

	def getCardDifference(self, considered_card, state):
		"""
		This method calculates the difference between the considered card and the highest card on the table
		args:
		considered_card : the card being considered
		state : the current state of the game
		
		returns:
		the difference between the considered card and the highest card on the table that is lower than the considered card
		"""
		highest_card_on_table = -1
		for row in state.rows:
			for card in row.cards:
				if card.number > highest_card_on_table:
					if card.number < considered_card.number:
						highest_card_on_table = card.number
		# if no such card exists, return the maximum value in the game : 104
		if highest_card_on_table == -1:
			return 104
		# return the difference between the two cards
		return abs(highest_card_on_table - considered_card.number)
	
	def getCardsPlayedInNearestRow(self, considered_card, state):
		"""
		This method calculates: 
		1. the number of cards played in the row where the considered card would go
		2. the penalty in that row
		args:
		considered_card : the card being considered
		state : the current state of the game

		returns:
		length : the number of cards played in the row where the considered card would go
		penalty : the penalty in that row
		"""
		# For the row where the considered card would go based on its value
		# find the number of cards that have been played in that row
		# get the last card of each row
		last_cards = [ row.cards[-1] for row in state.rows]
		# get cards that the considered card is greater than
		last_cards = [(last_cards[i],i) for i in range(len(last_cards)) if last_cards[i].number < considered_card.number]
		# if there is no such card, return 0
		if len(last_cards) == 0:
			return 0, 0
		# get the card that is closest to the considered card
		closest_card = min(last_cards, key=lambda x:abs(x[0].number - considered_card.number))
		# get the row where the closest card is
		length = len(state.rows[closest_card[1]].cards)
		penalty = state.rows[closest_card[1]].penalty()

		return length, penalty
	
	def getFeaturesForCardSelection(self, card, state):
		"""
		This method calculates the features for the card selection

		args:
		card : the card being considered
		state : the current state of the game

		returns:
		features : the features for the card selection
		"""
		average_penalty = self.totalpenalty
		if self.games > 0:
			average_penalty = self.totalpenalty / self.games
		value = card.number
		penalty = card.penalty
		# get the card difference
		card_diff = self.getCardDifference(card, state)
		# get the number of cards played in the nearest row and the penalty in that row
		cards_played_in_nearest_row, penalty_in_nearest_row = self.getCardsPlayedInNearestRow(card, state)
		# create the feature vector
		features = [value, penalty, card_diff, cards_played_in_nearest_row, penalty_in_nearest_row, average_penalty]
		return features

	def playCard(self, hand, rows, state, compute_next_state = True):
		"""
		This method chooses a card from the hand using epsilon greedy and updates the weights

		args:
		hand : the hand of the player
		rows : the rows on the table
		state : the current state of the game

		returns:
		hand[max_index] : the card with the maximum Q value
		"""
		Q_values = []
		for card in hand:
			# create the feature vector
			features = self.getFeaturesForCardSelection(card, state)
			# get the Q value for each action
			Q_value = [self.weights_play_card[i] * features[i] for i in range(len(features))][0]
			Q_values.append(Q_value)
		if random.uniform(0,1) < self.epsilon:
			max_index = random.randint(0,len(hand)-1)
			# current_state_features = self.getFeaturesForCardSelection(hand[max_index], state)
		else:
			for i in Q_values:
				if i == max(Q_values):
					max_index = Q_values.index(i)
			max_index = Q_values.index(max(Q_values))
		return hand[max_index]

	def chooseRow(self, rows, state):
		"""
		This method chooses a row from the rows on the table using epsilon greedy
		when exploring, choose a random row otherwise choose the row with the minimum penalty
		which is the hard coded strategy for the lowerPenaltyPlayer

		args:
		rows : the rows on the table
		state : the current state of the game

		returns:
		rowselect : the row to which the card will be added
		"""
		if random.uniform(0,1) < 0.1:
			return random.randint(0,len(rows)-1)
		# loop through each row and calculate Q(s,a)
		else:
			rowselect = 0
			for i in range(1,len(rows)):
				if rows[i].penalty() < rows[rowselect].penalty():
					rowselect = i
			return rowselect
		