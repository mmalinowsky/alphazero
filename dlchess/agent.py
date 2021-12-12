import random
import chess
import copy
import tensorflow.python.keras as keras
import tensorflow as tf

import numpy as np
from .encoder import Encoder, EncoderPlane
from dlchess.score import evaluate_board
import time

class Random:
	board = 0
	
	def __init__(self, board):
		self.board = board
	
	def set_color(self, player_color):
		self.player_color = player_color

	def move(self):
		return random.choice(list(self.board.legal_moves))



class MINMAX:
	board = 0
	states = {}
	start_time = 0
	player_color = 0


	def set_color(self, player_color):
		self.player_color = player_color

	def __init__(self, board):
		self.board = board

	def move(self):
		self.start_time = time.time()
		board = copy.deepcopy(self.board)
		[move, score] = self.best_move(board, 0)
		print('MINMAX SCORE' + str(score))
		return move

	def best_move(self, board, depth):
		MAXVAL = 9999
		temp_value = evaluate_board(board, self.player_color)
		if (temp_value == 9999) or (temp_value == -9999):
			return [board.peek(), temp_value]

		if depth > 4 or time.time() - self.start_time > 4:
			return [board.peek(), evaluate_board(board, self.player_color)]
		

		best_score = -9999
		legal_moves = (list(board.legal_moves))
		#random.shuffle(legal_moves)
		isort = []
		for i in legal_moves:
			board.push(i)
			isort.append((evaluate_board(board, self.player_color), i))
			board.pop()
		moves = sorted(isort, key=lambda x: x[0], reverse=board.turn)
		if depth > 1:
			moves = moves[0:3]
		move = random.choice(moves)[1]
		for i in [x[1] for x in moves]:
			#next_state = copy.deepcopy(board)
			#next_state.push(i)
			board.push(i)
			[new_move, opponent_max_value] = self.best_move(board, depth+1)
			board.pop()
			our_score = -1 * opponent_max_value
			#if our_score > 0:
				#print(our_score)
			if our_score > best_score:
				move = i
				best_score = our_score

		return [move, best_score]




class User:
	board = 0
	def __init__(self, board):
		self.board = board

	def move(self):
		move_input = input('Enter move:')
		move = chess.Move.from_uci(move_input)
		return move




class Zero:
	board = 0
	states = {}
	enc = Encoder()
	model = 0
	model_name = ""

	def __init__(self, board, model_name, apply_beam=True):
		self.board = board
		self.model = keras.models.load_model('models/'+model_name)
		self.model_name = model_name
		self.enc = EncoderPlane(12+1)
		self.apply_beam = apply_beam

	def set_color(self, player_color):
		self.player_color = player_color

	def indexToUci(self, index):
		return chess.SQUARE_NAMES[index]

	def getPredictions(self, player_color, board,):
		#if player_color == chess.WHITE:
		#	return self.getPredictions_white(board)

		return self.getPredictions_black(board)

	def getPredictions_black(self, board):
		encoded_board = self.enc.encode(board, board.turn)
		prediction = self.model.predict(np.array([encoded_board]))
		squares = []
		for c in range(8):
			for r in range(8):
				square_name = chess.SQUARE_NAMES[r+c*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
		return self.select_best_squares(squares)

	def getPredictions_white(self, board):
		encoded_board = self.enc.encode(board, board.turn)
		prediction = self.model.predict(np.array([encoded_board]))
		squares = []
		x = 0
		y = 0
		for c in range(7, 0, -1):
			x = 0
			for r in range(7, 0, -1):
				square_name = chess.SQUARE_NAMES[8-x+y*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
				x = x + 1
			y = y + 1
		return self.select_best_squares(squares)

	def select_best_squares(self, squares):
		ret = []
		heap_arr = [s['pred_value'] for s in squares]
		heapq.heapify(heap_arr)
		largest = heapq.nlargest(3, heap_arr)
		for i in largest:
			for s in squares:
				if s['pred_value'] == i:
					ret.append(s)
		return ret

	def select_moves(self, board, prediction_board):
		#print("["+self.model_name+"]max val=" + str(max(prediction_board[0])))
		squares = self.getPredictions(board.turn, board)
		moves_predicted = [chess.SQUARES[s['id']] for s in squares]
		#print(moves_predicted)
		#print(board.turn)
		moves = []
		#print(moves_predicted)
		for move in list(board.legal_moves):
			if move.to_square in moves_predicted:
				moves.append(move)
		return moves

	def move(self):
		encoded_board = self.enc.encode(self.board, self.board.turn)
		encoded_board = np.array([encoded_board])
		prediction_matrix = self.model.predict(encoded_board)
		moves = self.select_moves(self.board, prediction_matrix)
		print (moves)
		if not moves:
			print("No moves")
			input("Press to continue ")
			moves = self.board.legal_moves
		board = copy.deepcopy(self.board)
		move = moves[0]
		if self.apply_beam:
			[move, score] = self.best_move(board, 0, moves)
			print("Zero:", score)
		return move

	def beam(self, board, moves, amount):
		if type(moves) is not list:
			moves = [*moves]
		length = len(moves)
		#length = len(moves) if type(moves) is list else moves.count()
		diff = amount - length
		if diff > 0:
			isort = []
			for i in moves:
				if i not in board.legal_moves:
					continue
				board.push(i)
				isort.append((evaluate_board(board, self.player_color), i))
				board.pop()
			new_moves = sorted(isort, key=lambda x: x[0], reverse=board.turn)
			new_moves[0:diff]
			for move in new_moves:
				moves.append(move[1])
		return moves

	def best_move(self, board, depth, legal_moves):
		if depth > 6:
			return [board.peek(), evaluate_board(board, self.player_color)]
		if board.is_stalemate():
			return [board.peek(), 0]
		if board.is_checkmate():
			if board.turn != self.player_color:
				return [board.peek(), -9999]
			else:
				return [board.peek(), 9999]

		best_score = -9999
		#legal_moves = (list(board.legal_moves))
		#random.shuffle(legal_moves)
		#moves = self.beam(board, legal_moves, 4)
		moves = legal_moves
		#print(moves)
		if depth > 1:
			moves = moves[0:3]
		if len([*legal_moves]) > 0:
			move = random.choice([*legal_moves])
		elif len(moves) > 0:
			move = random.choice(moves)
		else:
			move = random.choice([*board.legal_moves])

		for i in moves:
			#next_state = copy.deepcopy(board)
			#next_state.push(i)
			if i not in board.legal_moves:
				continue
			board.push(i)
			#print(board)
			encoded_board = self.enc.encode(board, board.turn)
			encoded_board = np.array([encoded_board])
			pred_matrix = self.model.predict(encoded_board)
			predicted_moves = self.select_moves(board, pred_matrix)
			[new_move, opponent_max_value] = self.best_move(board, depth+1, predicted_moves)
			board.pop()
			our_score = -1 * opponent_max_value
			#if our_score > 0:
				#print(our_score)
			if our_score > best_score:
				move = i
				best_score = our_score

		return [move, best_score]


from collections import defaultdict
import math
import heapq


class MCTS:
	"Monte Carlo tree searcher. First rollout the tree then choose a move."

	def __init__(self, board, model, exploration_weight=1.4):
		self.Q = defaultdict(int)  # total reward of each node
		self.N = defaultdict(int)  # total visit count for each node
		self.children = dict()  # children of each node
		self.exploration_weight = exploration_weight
		self.board = board
		self.model = keras.models.load_model('models/'+model)
		self.enc = EncoderPlane(12+1)

	def set_color(self, player_color):
		self.player_color = player_color

	def move(self):
		self.Q = defaultdict(int)  # total reward of each node
		self.N = defaultdict(int)  # total visit count for each node
		self.children = dict()  # children of each node
		node = copy.deepcopy(self.board)
		board = copy.deepcopy(self.board)
		for _ in range(5):
			self.do_rollout(node)
		for c in self.children:
			print("visits", self.N[c], self.Q[c], (self.Q[c] / self.N[c]))

		
		ret = self.choose(self.board)
		print("chosen", self.N[ret], self.Q[ret])
		maxval = lambda n : float("-inf") if self.N[n] == 0 else ( float(self.Q[n]) / float(self.N[n]))
		uct = max(self.children[self.board], key=maxval)
		#print(uct)
		#print(self.N[uct], self.Q[uct])
		#print(self.N[self.board], self.Q[self.board])
		return ret.peek()

	def choose(self, node):
		"Choose the best successor of node. (Choose a move in the game)"
		if node.is_game_over():
			raise RuntimeError(f"choose called on terminal node {node}")

		if node not in self.children:
			cpy = copy.deepcopy(self.board)
			move = random.choice(list(node.legal_moves))
			cpy.push(move)
			return cpy

		def score(n):
			
			path = self._select(n)
			#z = len(path)
			#print(z, self.Q[n], self.N[n])
			if self.N[n] == 0:
				return float("-inf")  # avoid unseen moves
			return self.Q[n] / self.N[n]  # average reward
		for child in self.children:
			score(child)

		return max(self.children[node], key=score)

	def do_rollout(self, node):
		"Make the tree one layer better. (Train for one iteration.)"
		path = self._select(copy.deepcopy(node))
		leaf = path[-1]
		self._expand(leaf)
		leaf2 = copy.deepcopy(leaf)
		reward = self._simulate(leaf2)
		self._backpropagate(path, reward)

	def _select(self, node):
		"Find an unexplored descendent of `node`"
		path = []
		while True:
			path.append(node)
			if node not in self.children or not self.children[node]:
				# node is either unexplored or terminal
				return path
			unexplored = self.children[node] - self.children.keys()
			if unexplored:
				n = unexplored.pop()
				path.append(n)
				return path
			node = self._uct_select(node)  # descend a layer deeper

	def getPredictions(self, player_color, board,):
		#if player_color == chess.WHITE:
		#	return self.getPredictions_white(board)

		return self.getPredictions_black(board)

	def getPredictions_black(self, board):
		encoded_board = self.enc.encode(board, board.turn)
		prediction = self.model.predict(np.array([encoded_board]))
		squares = []
		for c in range(8):
			for r in range(8):
				square_name = chess.SQUARE_NAMES[r+c*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
		return self.select_best_squares(squares)

	def getPredictions_white(self, board):
		encoded_board = self.enc.encode(board, board.turn)
		prediction = self.model.predict(np.array([encoded_board]))
		squares = []
		x = 0
		y = 0
		for c in range(7, 0, -1):
			x = 0
			for r in range(7, 0, -1):
				square_name = chess.SQUARE_NAMES[8-x+y*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
				x = x + 1
			y = y + 1
		return self.select_best_squares(squares)

	def select_best_squares(self, squares):
		ret = []
		heap_arr = [s['pred_value'] for s in squares]
		heapq.heapify(heap_arr)
		largest = heapq.nlargest(3, heap_arr)
		for i in largest:
			for s in squares:
				if s['pred_value'] == i:
					ret.append(s)
		return ret

	def select_moves(self, board, prediction_board):
		#print("["+self.model_name+"]max val=" + str(max(prediction_board[0])))
		squares = self.getPredictions(board.turn, board)
		moves_predicted = [chess.SQUARES[s['id']] for s in squares]
		#print(moves_predicted)
		#print(board.turn)
		moves = []
		for move in list(board.legal_moves):
			if move.to_square in moves_predicted:
				moves.append(move)
		if len(moves) < 1:
			print("NO VALID MOVES FOUND")
			#print(board.turn)
			#img = chess.svg.board(board)
			#f = open('debug.html', 'w')
			#f.write(img)
			#f.close
			#input("A")
			return [*board.legal_moves]
		return moves

	def _expand(self, node):
		"Update the `children` dict with the children of `node`"
		if node in self.children:
			print("already expanded")
			return  # already expanded
		nodes = []
		encoded_board = self.enc.encode(node, node.turn)
		encoded_board = np.array([encoded_board])
		prediction_matrix = self.model.predict(encoded_board)
		#print(prediction_matrix)
		#print(node.turn)
		legal_moves = self.select_moves(node, prediction_matrix) 
		#legal_moves = [*node.legal_moves]
		for i in range(len(legal_moves)):
			nodes.append(copy.deepcopy(node))
			nodes[i].push(legal_moves[i])
		self.children[node] = nodes

	def _simulate(self, node):
		"Returns the reward for a random simulation (to completion) of `node`"
		#invert_reward = self.player_color == "White" ? True : False
		invert_reward = True
		to_reward = {"1/2-1/2" : 0, "0-1" : -1, "1-0" : 1}
		while True:
			if node.is_game_over():
				reward = node.result()
				reward = to_reward[reward]
				if reward == 0:
					return 0
				return reward if invert_reward else -(reward)
				#return -reward
				#return 1 - reward if invert_reward else reward
			encoded_board = self.enc.encode(node, node.turn)
			encoded_board = np.array([encoded_board])
			prediction_matrix = self.model.predict(encoded_board)
			moves = self.select_moves(node, prediction_matrix)
			move = random.choice(moves)
			#move = random.choice(list(node.legal_moves))
			node = copy.deepcopy(node)
			node.push(move)

			invert_reward = not invert_reward

	def _backpropagate(self, path, reward):
		"Send the reward back up to the ancestors of the leaf"
		for node in reversed(path):
			self.N[node] += 1
			self.Q[node] += reward
			if reward != 0:
				reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

	def _uct_select(self, node):
		"Select a child of node, balancing exploration & exploitation"

		# All children of node should already be expanded:
		#assert all(n in self.children for n in self.children[node])

		log_N_vertex = math.log(self.N[node])

		def uct(n):
			"Upper confidence bound for trees"
			return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
				log_N_vertex / self.N[n]
			)

		return max(self.children[node], key=uct)


class MyBoard(chess.Board):

	#def __init__():
		#self.board = board

	def __hash__(self):
		return hash(self.board_fen()+str(self.turn))
	def __eq__(self, other):
		return (
			self.__class__ == other.__class__ and
			(hash(self.board_fen()+str(self.turn)) == hash(other.board_fen()+str(other.turn))))
