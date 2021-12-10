import random
import chess
import copy
import tensorflow.python.keras as keras
import numpy as np
from dlchess.score import evaluate_board
from .encoder import EncoderPlane
import time

class Zero:
	board = 0
	states = {}
	enc = 0
	model = 0
	model_name = ""
	start_time = 0
	memory = []

	def __init__(self, board, model_name):
		self.board = board
		self.model = keras.models.load_model('models/'+model_name)
		self.enc = EncoderPlane(6*2+1)
		self.model_name = model_name
		self.max_depth = 5
	
	def indexToUci(self, index):
		return chess.SQUARE_NAMES[index]

	def move(self):
		self.start_time = time.time()
		encoded_board = self.enc.encode(self.board, self.board.turn)
		encoded_board = np.array([encoded_board])
		prediction_matrix = self.model.predict(encoded_board)
		item = np.random.choice(prediction_matrix[0], p=prediction_matrix[0])
		itemindex = np.where(prediction_board[0]==item)[0][0]
		legal_moves = self.board.legal_moves
		move_square = chess.SQUARES[itemindex]
		sample_moves.add(move_square)
		#moves = self.select_moves(self.board, prediction_matrix)
		#print (moves)
		#if not moves:
		#moves = self.board.pseudo_legal_moves
		#return moves[0]
		#return None
		board = copy.deepcopy(self.board)
		[move, score] = self.best_move(board, 0, moves)
		print('Alpha SCORE' + str(score) + " " + str(evaluate_board(board)))
		print(move)
		return move
