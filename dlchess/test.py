from generate_games import GameGenerator
import numpy as np

filename = "ficsgamesdb_202001_standard2000_nomovetimes_145127"
gen = GameGenerator()
[x, y] = gen.generate(filename)

#print(x.shape[0])
with open(filename+'.npy', 'wb') as f:
    np.save(f, x)
    np.save(f, y)
