import chess
import chess.svg
import dlchess.encoder as encoder
import numpy as np
import tensorflow.python.keras as keras
import heapq

board = chess.Board()
#model = keras.models.load_model('models/cnn_13p')
model = keras.models.load_model('models/cnn_13p_2021')
enc = encoder.EncoderPlane(12+1)


def printPredictions(board, enc, model):
	encoded_board = enc.encode(board, board.turn)
	prediction = model.predict(np.array([encoded_board]))
	rowMsg = []
	for c in range(8):
		for r in range(8):
			rowMsg.append(str('{:06.2f}'.format(prediction[0][r+(c*8)]))+" ")
		print(rowMsg)
		rowMsg = []

def getPredictions(board, enc, model):
	encoded_board = enc.encode(board, board.turn)
	prediction = model.predict(np.array([encoded_board]))
	squares = []
	for c in range(8):
		for r in range(8):
			if prediction[0][r+c*8] > 0.02:
				square_name = chess.SQUARE_NAMES[r+c*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
	return squares


while(True):
	player_turn = board.turn
	print("player_turn", player_turn)
	printPredictions(board, enc, model)
	squares = getPredictions(player_turn, board, enc,model)
	print(squares)

	s_set = chess.SquareSet([s['id'] for s in squares])
	img = chess.svg.board(board, squares=s_set)
	f = open('test.html', 'w')
	f.write(img)
	f.close
	encoded_board = enc.encode(board, board.turn)
	#print(encoded_board)
	#print("enc")
	prediction_board = model.predict(np.array([encoded_board]))
	print(prediction_board)
	item = np.random.choice(prediction_board[0], p=prediction_board[0])
	itemindex = np.where(prediction_board[0]==item)[0][0]
	legal_moves = board.legal_moves
	move_square = chess.SQUARES[itemindex]
	#print(move_square)
	moves = [chess.SQUARES[s['id']] for s in squares]
	for move in list(board.legal_moves):
		if move.to_square in moves:
			print("Found valid move", move)
	move_input = input('Enter move:')
	move = chess.Move.from_uci(move_input)
	board.push(move)
