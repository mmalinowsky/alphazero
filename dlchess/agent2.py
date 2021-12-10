import random
import chess
import copy
import keras
import numpy as np
from encoder import Encoder

def evaluate_board(board):
    
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)
    eval = material
    if board.turn:
        return eval
    else:
        return -eval

class Random:
	board = 0
	def __init__(self, board):
		self.board = board
	def move(self):
		return random.choice(list(self.board.legal_moves))



class MCTS:
	board = 0
	states = {}
	def __init__(self, board):
		self.board = board

	def move(self):
		board = copy.deepcopy(self.board)
		[move, score] = self.best_move(board, 0)
		print('MCTS SCORE' + str(score))
		return move

	def best_move(self, board, depth):

		self.states[board.unicode()] = 1
		if depth > 4:
			return [board.peek(), evaluate_board(board)]
		if board.is_stalemate():
			return [board.peek(), 0]
		if board.is_checkmate():
			if board.turn:
				return [board.peek(), -9999]
			else:
				return [board.peek(), 9999]

		best_score = -9999
		legal_moves = (list(board.legal_moves))
		#random.shuffle(legal_moves)
		isort = []
		for i in legal_moves:
			board.push(i)
			isort.append((evaluate_board(board), i))
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
	model = keras.models.load_model('models/first')
	enc = Encoder()
	def __init__(self, board):
		self.board = board

	def indexToUci(self, index):
		return chess.SQUARE_NAMES[index]

	def select_moves(self, prediction_board):
		item = np.random.choice(prediction_board[0], p=prediction_board[0])
		itemindex = np.where(prediction_board[0]==item)[0][0]
		legal_moves = self.board.legal_moves
		move_square = chess.SQUARES[itemindex]
		moves = []
		for move in legal_moves:
			#print(str(move.to_square) + " " + str(move_square))
			if move.to_square == move_square:
				moves.append(move)
		#print(moves)
		#print(self.indexToUci(itemindex))
		return moves
	def move(self):
		encoded_board = self.enc.encode(self.board, self.board.turn)
		encoded_board = encoded_board.reshape(1,64)
		prediction_matrix = self.model.predict(encoded_board)
		moves = self.select_moves(prediction_matrix)
		print (moves)
		if not moves:
			moves = self.board.legal_moves
		#return moves[0]
		#return None
		board = copy.deepcopy(self.board)
		[move, score] = self.best_move(board, 0, moves)
		#print('MCTS SCORE' + str(score))
		return move

	def best_move(self, board, depth, legal_moves):

		self.states[board.unicode()] = 1
		if depth > 4:	
			return [board.peek(), evaluate_board(board)]
		if board.is_stalemate():
			return [board.peek(), 0]
		if board.is_checkmate():
			if board.turn:
				return [board.peek(), -9999]
			else:
				return [board.peek(), 9999]

		best_score = -9999
		#legal_moves = (list(board.legal_moves))
		#random.shuffle(legal_moves)
		isort = []
		for i in legal_moves:
			board.push(i)
			isort.append((evaluate_board(board), i))
			board.pop()
		moves = sorted(isort, key=lambda x: x[0], reverse=board.turn)
		if depth > 1:
			moves = moves[0:3]
		move = random.choice(moves)[1]
		for i in [x[1] for x in moves]:
			#next_state = copy.deepcopy(board)
			#next_state.push(i)
			board.push(i)
			[new_move, opponent_max_value] = self.best_move(board, depth+1, board.legal_moves)
			board.pop()
			our_score = -1 * opponent_max_value
			#if our_score > 0:
				#print(our_score)
			if our_score > best_score:
				move = i
				best_score = our_score

		return [move, best_score]