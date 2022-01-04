import chess.pgn
import dlchess.encoder as encoder
import numpy as np


class GameGenerator:

	def get_winner_color(self, header):
		color_map = { 
		"1/2-1/2" : 0,
		"0-1" : 1,
		"1-0" : 0}
		return color_map[header]


	def generate(self, filename, planes, games_count=1000):

		enc = encoder.EncoderPlane(planes)
		pgn = open(filename+".pgn")
		game_results = ["1/2-1/2", "0-1", "1-0"]
		game = chess.pgn.read_game(pgn)
		X = []
		Y = []
		games = 0
		moves_by_color = {0: 0, 1: 0}
		while (game is not None):
			board = game.board()
			winner_color = self.get_winner_color(game.headers['Result'])
			#if game.headers['Result'] != game_results[1]:
				#game = chess.pgn.read_game(pgn)
				#continue
			if game.headers['Result'] not in game_results:
				continue

			for move in game.mainline_moves():
				if board.turn != bool(winner_color):
					board.push(move)
					continue
				moves_by_color[board.turn] = moves_by_color[board.turn] + 1
				x = enc.encode(board, board.turn)
				X.append(x)
				y = enc.move_encode(move)
				Y.append(y)
				board.push(move)

			if games > games_count:
				break
			#print(np.array(Y))
			#print(Y.size)
			#print(Y.shape)
			#print(Y.shape)
			games = games + 1
			game = chess.pgn.read_game(pgn)
		Y = np.array(Y)
		print(moves_by_color)
		return [np.array(X), Y]
