from flask import Flask,jsonify

import chess
import chess.svg
import dlchess.agent as agent
from threading import Thread

colors = ['Black', 'White']

opponent_color = chess.WHITE



__all__ = [
    'get_web_app',
]
js_data = '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>\
<script type="text/javascript">\
$("use").click(function(el) { \
	var a = el['currentTarget']['attributes'][1].nodeValue;\
	console.log(a);\
	v = a.split('(')[1];\
    w = v.split(')')[0];\
    arr = v.split(',');\
\
    x = parseInt(arr[0]);\
    y = parseInt(arr[1]);\
    id = Math.floor(x/45) + (Math.floor(y/45) * 8);\
    console.log(id);\
\
});\
</script>\
'


def get_web_app(opponent, board):
	app = Flask(__name__)
	opponent.player_color = opponent_color
	def opponent_move(agent, board):
		move = agent.move()
		board.push(move)

	@app.route('/')
	def index():
		if board.is_game_over():
			return 'Game is over'

		return js_data+'<div id="result"></div><button onclick="loadAsset(\'next\', \'json\', displayImage)">Move</button> <div width="50px"> Turn:' + str(colors[board.turn]) + '<div id="board_svg">' + chess.svg.board(board=board, size=640) +'</div></div>'

	@app.route('/next')
	def next():
		if board.turn is opponent_color:
			opponent_move(opponent, board)
		return jsonify({'done': '1'})

	@app.route('/move/<string:move_uci>')
	def move(move_uci):
		if board.is_game_over():
			return 'Cant move'
		if board.turn is opponent_color:
			#board.push(opponent.move())
			#thread = Thread(target = opponent_move, args = (opponent, board))
			#thread.start()
			#thread.join()
			return 'Not your turn'
		move = chess.Move.from_uci(move_uci)
		if move in board.pseudo_legal_moves:
			board.push(move)
		return index()

	return app
