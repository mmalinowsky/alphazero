import chess
import dlchess.encoder as encoder
import numpy as np


board = chess.Board()

enc = encoder.EncoderPlane(13)

result = enc.encode(board, board.turn)
print(board.turn - 0)
print(len(result))
print(result)

print("new Move")

move = [*board.legal_moves][0]
board.push(move)
print(board.turn - 0)
result = enc.encode(board, board.turn)

print(len(result))
print(result)
