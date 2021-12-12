import chess
import chess.svg
import copy
from dlchess.agent import *
from dlchess.alpha import Alpha
import time
from enum import Enum

board = MyBoard()
Players = {
	#chess.WHITE: MCTS(board, "cnn_13p_2021"),
	#chess.BLACK: Zero(board, "cnn_13p"),
	#chess.WHITE: agent.User(board),
	chess.WHITE: Zero(board, "cnn_13p_2021"),
	#chess.BLACK: Zero(board, "cnn_13p_2021"),
	chess.BLACK: MINMAX(board),
	#chess.WHITE: Alpha(board, "cnn_13p"),
	#chess.BLACK: Random(board)
}

for k in Players:
	Players[k].set_color(k)

round = 0
while not board.is_game_over():
	#Players[chess.WHITE] = MCTS(board)
	#Players[chess.BLACK] = MINMAX(board)
	start = time.time()
	#print('Score:'+str(current)+' '+str(agent.evaluate_board(board)))
	current_color = board.turn
	move = Players[current_color].move()
	board.push(move)

	img = chess.svg.board(board=board)
	f = open('svg/test'+str(round)+'.html', 'w')
	f.write(img)
	f.close
	round += 1
	print("elapsed:" + str(time.time()-start))

print(board)
print(board.result())

