import chess

from dlchess.agent import *
from dlchess import serve
from dlchess.alpha import Alpha


def main():

	board = MyBoard()
	#bot =  MCTS(board, 'cnn_13p')
	#bot = Zero(board, "cnn_13p"),
	bot = Random(board)
	web_app = serve.get_web_app(bot, board)
	web_app.run(host="localhost", port=5000, threaded=False, debug=True)


if __name__ == '__main__':
    main()

