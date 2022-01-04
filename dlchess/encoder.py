import numpy as np
import chess
import math


class EncoderPlane:
	
	def __init__(self, planes=0):
		self.size = 8
		self.shape = (planes, 8, 8) if planes > 0 else (8,8)
		self.planes = planes

	def plane_features(self, feature_id, board, player_color):
		plane = np.zeros(self.size * self.size, dtype=int)
		squares = board.pieces(feature_id, player_color)
		for sq in squares:
			plane[sq] = 1
		plane = np.resize(plane, (self.size, self.size))
		return plane

	def encode(self, board, player_color):
		encoded_board = np.zeros(self.shape, dtype=int)

		pieces = board.piece_map()
		enemy_color = chess.WHITE if player_color == chess.BLACK else chess.BLACK
		planes_num = math.floor((self.planes-1)/2)
		for c in [player_color, enemy_color]:
			for p in range(1, planes_num+1):
				index = c*planes_num+p-1
				encoded_board[index] = self.plane_features(p, board, c)
		board_color = np.zeros((self.size, self.size), dtype=int)
		board_color.fill(board.turn)
		encoded_board[self.planes-1] = board_color
		#np.append(encoded_board, board_color)
		return encoded_board
		for pos in pieces:
			y = math.floor(pos / self.size)
			x = pos - (y * self.size)
			if pieces[pos].color == player_color:
				encoded_board[0][y][x] = 1
			else:
				encoded_board[0][y][x] = -1
		return encoded_board

	def move_encode(self, move):
		encoded_board = np.zeros((self.size * self.size), dtype=int)
		sq = move.to_square
		encoded_board[sq] = 1
		#encoded_board = np.reshape(encoded_board, (self.size, self.size))
		return encoded_board