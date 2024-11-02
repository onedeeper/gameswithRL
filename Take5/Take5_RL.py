from Take5Player import Player
from Take5State import State, PlayerState
from Take5Game import Card, Row
from Take5 import Game
from Take5 import Play
from collections import defaultdict
import random
import copy
import numpy as np
import pickle

class RLAgent(Player):
	"""
	Implements a reinforcement learning agent that learns to play the game of Take5 using
	Linear function approximation with Q-learning

	NOTE : 
	There was major refactoring done to the Take5.py file to make it work with the RLAgent during training
	for example, the simulate round method requires an object which deviates from the original implementation 
	of the state representation.
	"""
	Qtable = defaultdict(float)
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
		self.epsilon = 0.9
		self.alpha = 0.00001
		self.gamma = 0.8
		self.weights_play_card = np.random.normal(0,1,6)
		self.train = False
		if not self.train:
			self.epsilon = 0
			# load the weights from the pickle file
			with open('1_mill_.pickle', 'rb') as f:
				self.weights_play_card = pickle.load(f)

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
		features = np.array([value, penalty, card_diff, cards_played_in_nearest_row, penalty_in_nearest_row, average_penalty])
		return features

	def weightUpdateCard(self, reward, Q_value_for_chosen_card, current_state_features, max_Q_value_for_next_state = None):
		"""
		This method updates the weights for the card selection using the temporal difference error update rule

		args:
		reward : the reward for the action
		Q_value_for_chosen_card : the Q value for the chosen card
		current_state_features : the features for the current state
		max_Q_value_for_next_state : the maximum Q value for the next state

		returns:
		None
		"""
   		# Compute the temporal difference error
		if max_Q_value_for_next_state is not None:
			# If we're not at a terminal state, include the discounted future Q value
			temporal_difference_error = (reward + self.gamma * max_Q_value_for_next_state) - Q_value_for_chosen_card
		else:
			# If we're at a terminal state, there is no future Q value
			temporal_difference_error = reward - Q_value_for_chosen_card

		# Update the weights
		self.weights_play_card += self.alpha * temporal_difference_error * current_state_features

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
		# decrease epsilon by 0.001 every 1000 games after 100000 games
		if self.games > 900000:
			self.epsilon -= 0.0001
			self.epsilon = max(self.epsilon, 0)
			#print(self.epsilon)
		penalty_in_hand = self.penalty()
		Q_values = []
		for card in hand:
			# create the feature vector
			features = self.getFeaturesForCardSelection(card, state)
			# get the Q value for each action
			Q_value = np.dot(self.weights_play_card, features)
			Q_values.append(Q_value)
		if random.uniform(0,1) < self.epsilon:
			max_index = random.randint(0,len(hand)-1)
			current_state_features = self.getFeaturesForCardSelection(hand[max_index], state)
		else:
			max_index = np.argmax(Q_values)
			current_state_features = self.getFeaturesForCardSelection(hand[max_index], state)
		if self.train:
			# get the next state after the action
			next_state = self.getNewStateCard(state, hand[max_index])
			# next state features for the chosen card
			next_state_features = self.getFeaturesForCardSelection(hand[max_index], next_state)
			# if there are no cards in the hand after the action, the game is over so calculate the immediate reward
			for player in next_state.selectedplayers:
				if player.name == "RLAgent" and len(player.hand) == 0:
					# penalty of hand after the action
					penalty_in_hand_next_state = player.penalty()
					# get the reward
					reward = self.getRewardCard(penalty_in_hand_next_state, penalty_in_hand, next_state_features[-2], next_state_features[-1])
					# update the weights
					self.weightUpdateCard(reward, Q_values[max_index],current_state_features, None)
					return hand[max_index]
			# calculate reward for each possible action in the next state
			Q_value_for_cards_in_next_state = []
			for player in next_state.selectedplayers:
				if player.name == 'RLAgent':
					penalty_in_hand_next_state = player.penalty()
					for card in player.hand:
						features = self.getFeaturesForCardSelection(card, next_state)
						Q_value = np.dot(self.weights_play_card, features)
						Q_value_for_cards_in_next_state.append(Q_value)		
			# get the reward
			reward = self.getRewardCard(penalty_in_hand_next_state, penalty_in_hand, next_state_features[-2], next_state_features[-1])
			# update the weights
			self.weightUpdateCard(reward, Q_values[max_index],current_state_features,np.max(Q_value_for_cards_in_next_state) )
		# return the card with the maximum Q value
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
			
	def getRewardCard(self, penalty_next_hand, penalty_previous_hand, length_of_nearest_row, penalty_in_nearest_row):
		"""
		This method calculates the reward for the action

		args:
		penalty_next_hand : the penalty of the hand after the action
		penalty_previous_hand : the penalty of the hand before the action
		length_of_nearest_row : the number of cards played in the row where the considered card would go
		penalty_in_nearest_row : the penalty in that row

		returns:
		reward : the reward for the action
		"""
		# positive reward if the penalty in the hand decreases
		reward = penalty_previous_hand #- penalty_next_hand
		# if the nearest row has 5 cards already, the row must be taken
		if length_of_nearest_row == 5:
			reward = -(penalty_in_nearest_row)
		return reward 
	
	def simulateRound(self, new_card, selectedplayers, number, state, compute_next_state = True):
		"""
		This method simulates one round of the game and returns the new state.
		Basically the same as the playRound method in Take5.py with some changes to retain a copy of the state
		so that nothing is changed in the original state.

		args:
		new_card : the card being considered
		selectedplayers : the players playing the game
		number : the round number
		state : the current state of the game

		returns:
		new_state : the new state of the game
		"""
		# Simulate one round of the game. 
		# 
		plays = []
		state = copy.deepcopy(state)
		selectedplayers = copy.deepcopy(selectedplayers)
		state = State( number, state.rows, selectedplayers, selectedplayers )
		for player in selectedplayers:
			if player.name != 'RLAgent':
				card = player.playCard( player.hand, state.rows, state )
			if player.name == 'RLAgent':
				card = new_card
			if card not in player.hand:
				card = player.hand[random.randrange( len( player.hand ) )]
			player.lastcard = card
			plays.append( Play( player, card ) )
		plays.sort()
		for play in plays:
			play.player.removeCard( play.card )
			rowselect = play.card.goesToRow( state.rows )
			if rowselect >= 0 and len( state.rows[rowselect].cards ) < 5:
				state.rows[rowselect].addCard( play.card )
			else:
				if rowselect < 0:
					rowselect = play.player.chooseRow( state.rows, state )
				if rowselect < 0 or rowselect > 3:
					rowselect = random.randrange( 4 )
				for card in state.rows[rowselect].cards:
					play.player.addCardToCollection( card )
				state.rows[rowselect].startNewRow( play.card )
		
		return {'number' : number, 'rows' : state.rows , 'plays' : plays}
	
	def getNewStateCard(self,state, card, compute_next_state = True):
		# get the new state after the action
		new_state = self.simulateRound(card, state.selectedplayers, state.round + 1, state, compute_next_state)
		players = []
		for play in new_state['plays']:
			play.player.games += 1
			play.player.totalpenalty += play.player.penalty()
			players.append(play.player)
		new_state = State(state.round + 1, new_state['rows'], players, players)
		return new_state