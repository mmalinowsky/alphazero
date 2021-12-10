import chess
import chess.svg
import dlchess.encoder as encoder
import numpy as np
import tensorflow.python.keras as keras

board = chess.Board()
model = keras.models.load_model('models/cnn_7p')
enc = encoder.EncoderPlane(6*2)



def printPredictions(board, enc, model):
	encoded_board = enc.encode(board, board.turn)
	prediction = model.predict(np.array([encoded_board]))
	rowMsg = []
	for c in range(8):
		for r in range(8):
			rowMsg.append(str('{:06.2f}'.format(prediction[0][r+c*8]))+" ")
		print(rowMsg)
		rowMsg = []

def getPredictions(player_color, board, enc, model):
	if player_color == chess.WHITE:
		return getPredictions_white(board,enc,model)

	return getPredictions_black(board,enc,model)

def getPredictions_black(board, enc, model):
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

def getPredictions_white(board, enc, model):
	encoded_board = enc.encode(board, board.turn)
	prediction = model.predict(np.array([encoded_board]))
	squares = []
	x = 0
	y = 0
	for c in range(7, 0, -1):
		x = 0
		for r in range(7, 0, -1):
			if prediction[0][r+c*8] > 0.02:
				square_name = chess.SQUARE_NAMES[8-x+y*8]
				square = chess.parse_square(square_name)
				square_obj = {'name': square_name, 'id': square, 'pred_value':prediction[0][r+c*8]}
				squares.append(square_obj)
			x = x + 1
		y = y + 1
	return squares


while(True):
	player_turn = board.turn
	printPredictions(board, enc, model)
	squares = getPredictions(player_turn, board, enc,model)
	print(squares)

	s_set = chess.SquareSet([s['id'] for s in squares])
	img = chess.svg.board(board, squares=s_set)
	f = open('test.html', 'w')
	f.write(img)
	f.close
	encoded_board = enc.encode(board, board.turn)
	prediction_board = model.predict(np.array([encoded_board]))
	item = np.random.choice(prediction_board[0], p=prediction_board[0])
	itemindex = np.where(prediction_board[0]==item)[0][0]
	legal_moves = board.legal_moves
	move_square = chess.SQUARES[itemindex]
	#print(move_square)
	legal_moves = list(board.legal_moves)
	moves = [chess.SQUARES[s['id']] for s in squares]
	for move in legal_moves:
		if move.to_square in moves:
			print("GITUWA!!", move)
	move_input = input('Enter move:')
	move = chess.Move.from_uci(move_input)
	board.push(move)
