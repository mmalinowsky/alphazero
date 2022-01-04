import chess
import chess.svg
import copy
from dlchess.agent import *
from dlchess.alpha import Alpha
import time
from enum import Enum

board = MyBoard()
Players = {
	chess.BLACK: MCTS(board, "big"),
	#chess.BLACK: Zero(board, "cnn_13p"),
	#chess.WHITE: agent.User(board),
	#chess.BLACK: Zero(board, "big"),
	#chess.BLACK: Zero(board, "cnn_13p_2021"),
	chess.WHITE: MINMAX(board),
	#chess.WHITE: Alpha(board, "cnn_13p"),
	#chess.BLACK: Random(board)
}

for k in Players:
	Players[k].set_color(k)

round = 0
output_svg = True

while not board.is_game_over():
	start = time.time()
	#print('Score:'+str(current)+' '+str(agent.evaluate_board(board)))
	current_color = board.turn
	move = Players[current_color].move()
	board.push(move)
	if output_svg:
		img = chess.svg.board(board=board)
		f = open('svg/test'+str(round)+'.html', 'w')
		f.write(img)
		f.close
	round += 1
	print("elapsed:" + str(time.time()-start))

print(board.result())

