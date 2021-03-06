from dlchess.game_generator import GameGenerator
import numpy as np

filename = "ficsgamesdb_202001_standard2000_nomovetimes_145127"
gen = GameGenerator()
[x, y] = gen.generate("pgn/"+filename, 6*2+1)
#print(x.shape[0])
with open("games/1000games.npy", 'wb') as f:
    np.save(f, x)
    np.save(f, y)
