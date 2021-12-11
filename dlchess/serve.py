from flask import Flask, jsonify ,render_template, make_response
import os
import chess
import chess.svg
import dlchess.agent as agent
from threading import Thread

colors = ['Black', 'White']


__all__ = [
    'get_web_app',
]


def get_web_app(opponent, board):
	app = Flask(__name__,
		static_folder='web/static',
		static_url_path='', 
        template_folder='web/templates'
		)

	#opponent.player_color = opponent_color

	def opponent_move(agent, board):
		move = agent.move()
		board.push(move)

	@app.route('/')
	def index():
		title = 'Chess'
		return render_template('index.html', title=title)

	@app.route('/board_svg')
	def board_svg():
		return chess.svg.board(board=board, size=640)

	@app.route('/next')
	def next():
		if board.turn is opponent.player_color:
			opponent_move(opponent, board)
		return jsonify({'done': '1'})

	@app.route('/move/<string:move_uci>')
	def move(move_uci):
		if board.is_game_over():
			return jsonify({'error': 'Game is Over'})
		if board.turn is opponent.player_color:
			#board.push(opponent.move())
			#thread = Thread(target = opponent_move, args = (opponent, board))
			#thread.start()
			#thread.join()
			return jsonify({'error': 'Not your turn'})
		move = chess.Move.from_uci(move_uci)
		if move in board.pseudo_legal_moves:
			board.push(move)
			return jsonify({'result': 'Success'})
		return jsonify({'error': move_uci + ' is not a valid move'})

	return app
