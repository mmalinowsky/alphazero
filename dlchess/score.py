import chess

def evaluate_board(board, player_color):
	if board.is_game_over():
		if board.result() == "1-0":
			return 9999 if player_color == 1 else -9999
		elif board.result() == "0-1":
			return -9999 if player_color == 1 else 9999
		else:
			return 0

	if board.is_stalemate():
		return 0

	wp = len(board.pieces(chess.PAWN, chess.WHITE))
	bp = len(board.pieces(chess.PAWN, chess.BLACK))
	wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
	bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
	wb = len(board.pieces(chess.BISHOP, chess.WHITE))
	bb = len(board.pieces(chess.BISHOP, chess.BLACK))
	wr = len(board.pieces(chess.ROOK, chess.WHITE))
	br = len(board.pieces(chess.ROOK, chess.BLACK))
	wq = len(board.pieces(chess.QUEEN, chess.WHITE))
	bq = len(board.pieces(chess.QUEEN, chess.BLACK))

	eval = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)
	if board.turn:
		return eval
	else:
		return -eval
