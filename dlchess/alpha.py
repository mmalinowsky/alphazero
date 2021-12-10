import random
import chess
import copy
import tensorflow.python.keras as keras
import numpy as np
from dlchess.score import evaluate_board
from .encoder import EncoderPlane
import time

class Alpha:
	board = 0
	states = {}
	enc = 0
	model = 0
	model_name = ""
	start_time = 0
	def __init__(self, board, model_name):
		self.board = board
		self.model = keras.models.load_model('models/'+model_name)
		self.enc = EncoderPlane(6*2+1)
		self.model_name = model_name
		self.max_depth = 5
	def indexToUci(self, index):
		return chess.SQUARE_NAMES[index]

	def select_moves(self, board, prediction_board):
		#print("["+self.model_name+"]max val=" + str(max(prediction_board[0])))
		remaining_moves = 3
		sample_moves = set()
		while remaining_moves > 0:
			item = np.random.choice(prediction_board[0], p=prediction_board[0])
			itemindex = np.where(prediction_board[0]==item)[0][0]
			#legal_moves = self.board.legal_moves
			move_square = chess.SQUARES[itemindex]
			sample_moves.add(move_square)
			remaining_moves -=1
		moves = []
		for move in board.pseudo_legal_moves:
			#print(str(move.to_square) + " " + str(move_square))
			if move.to_square in sample_moves:
				moves.append(move)
		#print(moves)
		#print(self.indexToUci(itemindex))
		return moves

	def move(self):
		self.start_time = time.time()
		encoded_board = self.enc.encode(self.board, self.board.turn)
		encoded_board = np.array([encoded_board])
		prediction_matrix = self.model.predict(encoded_board)
		moves = self.select_moves(self.board, prediction_matrix)
		#print (moves)
		#if not moves:
		moves = self.board.pseudo_legal_moves
		#return moves[0]
		#return None
		board = copy.deepcopy(self.board)
		[move, score] = self.best_move(board, 0, moves)
		print('Alpha SCORE' + str(score) + " " + str(evaluate_board(board)))
		print(move)
		return move

	def beam(self, board, moves, amount):
		if type(moves) is not list:
			moves = [*moves]
		length = len(moves)
		#length = len(moves) if type(moves) is list else moves.count()
		diff = amount - length
		result = []
		isort = []
		for i in moves:
			if i in board.pseudo_legal_moves:
				board.push(i)
				isort.append((evaluate_board(board), i))
				board.pop()
		new_moves = sorted(isort, key=lambda x: x[0], reverse=board.turn)
		new_moves[0:amount]
		for move in new_moves:
			result.append(move[1])
		return result

	def best_move(self, board, depth, legal_moves):
		MAXVAL = 9999
		if board.is_game_over():
			if board.result() == "1-0":
				return [board.peek(), MAXVAL]
			elif board.result() == "0-1":
				return [board.peek(), -MAXVAL]
			else:
				return [board.peek(),0]
		if time.time() - self.start_time > 4:
			return [board.peek(), evaluate_board(board)]
		if board.is_stalemate():
			return [board.peek(), 0]
		
		#if board.is_checkmate():
		#	if board.turn:
		#		return [board.peek(), -9999]
		#	else:
		#		return [board.peek(), 9999]

		best_score = -9999
		opponent_max_value = 0
		#legal_moves = (list(board.legal_moves))
		#random.shuffle(legal_moves)
		moves = self.beam(board, legal_moves, 6)
		
		#moves = moves[0:moves_len]
		if len(moves) < 1:
			return [board.peek(), evaluate_board(board)]
		#if len(moves) < 1:
		#	print("test")
		#	return [board.peek(), evaluate_board(board)]
		#print(moves)
		#if depth > 1:
			#moves = moves[0:6]
		#if len([*legal_moves]) > 0:
		#	move = random.choice([*legal_moves])
		#else:
		#if len(moves) < 1:
			#moves = [*board.legal_moves]
			#print([*board.legal_moves])
		#	best = self.beam(board, [*board.pseudo_legal_moves], 8)
			#print("best")
			#print(best)
		#	return [best[0], evaluate_board(board)]
		move = moves[0]
		for i in moves:
			#next_state = copy.deepcopy(board)
			#next_state.push(i)
			board.push(i)
			encoded_board = self.enc.encode(board, board.turn)
			encoded_board = np.array([encoded_board])
			prediction_matrix = self.model.predict(encoded_board)
			moves2 = self.select_moves(board, prediction_matrix)
			#if len(moves2) < 1:
				#moves2 = board.legal_moves
			#if len(beamed) > 1:
			[new_move, opponent_max_value] = self.best_move(board, depth+1, moves2)
			board.pop()
			our_score = -1 * opponent_max_value
			#if our_score > 0:
				#print(our_score)
			if our_score > best_score:
				move = i
				#print("new best score")
				#print(our_score)
				#print(new_move)
				#if new_move in board.legal_moves:
					#move = new_move
				best_score = our_score

		return [move, best_score]
