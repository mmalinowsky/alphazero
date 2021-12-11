import chess

from dlchess.agent import *
from dlchess import serve
from dlchess.alpha import Alpha



def main():

	board = MyBoard()
	bot = MCTS(board, "cnn_13p")
	opponent_color = chess.BLACK
	bot.set_color(opponent_color)
	web_app = serve.get_web_app(bot, board)
	web_app.run(host="localhost", port=5000, threaded=False)


if __name__ == '__main__':
    main()

