# AlphaZero
Trying to recreate AlphaZero for chess using this paper
[https://arxiv.org/pdf/1712.01815.pdf](https://arxiv.org/pdf/1712.01815.pdf)


## Getting started
In order to play you have to train models first so run one of the following scripts from the directory:

```
python3 train.py
```

or for self-play training 
```
python3 a2c_train.py
```

To start web based chess play type

```
python3 web.py
```

## Docker installation

```
docker build -t chess
```

