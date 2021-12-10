import chess.pgn
import dlchess.encoder as encoder
import numpy as np


class GameGenerator:

	def get_winner_color(header):
		color_map = { 
		"1/2-1/2" : 0,
		"0-1" : 1,
		"1-0" : 0}
		return color_map[header]

	def generate(self, filename, planes):

		enc = encoder.EncoderPlane(planes)
		pgn = open(filename+".pgn")
		game_results = ["1/2-1/2", "0-1", "1-0"]
		game = chess.pgn.read_game(pgn)
		X = []
		Y = []
		#current_move_color = 0
		winner_color = self.get_winner_color(game.headers['Result'])
		while (game is not None):
			board = game.board()

			if game.headers['Result'] not in game_results:
				continue
			for move in game.mainline_moves():
				if board.turn != winner_color:
					continue
				x = enc.encode(board, board.turn)
				X.append(x)
				y = enc.move_encode(move)
				Y.append(y) 
				board.push(move)
			#print(np.array(Y))
			#print(Y.size)
			#print(Y.shape)
			
			#print(Y.shape)
			game = chess.pgn.read_game(pgn)
		Y = np.array(Y)
		return [np.array(X), Y]
