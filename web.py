import chess

from dlchess.agent import *
from dlchess import serve
from dlchess.alpha import Alpha
import argparse

bots = {"zero" : Zero, "minmax" : MINMAX, "random": Random, "mcts": MCTS}

def main():
	board = MyBoard()
	parser = argparse.ArgumentParser(description='Web chess game')
	parser.add_argument('--bot', metavar='bot_name', nargs='?', type=str, required=True, help='Bot class name')
	args = parser.parse_args()
	if not args.bot in bots:
		exit("Bot named " + str(args.bot) + "not found")
	bot = bots[args.bot](board, "latest_model")
	opponent_color = chess.BLACK
	bot.set_color(opponent_color)
	web_app = serve.get_web_app(bot, board)
	web_app.run(host="localhost", port=5000, threaded=False)


if __name__ == '__main__':
    main()

