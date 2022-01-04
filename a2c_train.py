import os.path
import numpy as np
from tensorflow import keras
import tensorflow as tf
import chess
import chess.svg
import copy
from dlchess.agent import *
import time
from enum import Enum
from dlchess.score import evaluate_board
from dlchess.model import AC_model

model_filepath = 'models/latest_model.h5'
planes = 6 * 2 + 1
batch_size = 128
epochs = 100


input_shape = (planes, 8, 8)

np.random.seed(123)



class model_wrapper():

	def __init__(self):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
		self.huber_loss = tf.keras.losses.Huber()
		self.enc = EncoderPlane(12+1)
		self.model = model = AC_model()
		self.model.build(input_shape=(1, planes, 8, 8))
		self.reset()
	
	def reset(self):
		self.action_probs_history = []
		self.critic_value_history = []
		self.rewards_history = []
		self.running_reward = 0
		self.episode_count = 0
		self.episode_reward = 0

	def new_game(self, board):
		self.board = board
		self.opponent = Random(self.board)
		#self.opponent = MINMAX(self.board)
		#self.opponent.set_color = chess.BLACK

	def get_best_move(self, square):
		moves = []
		for move in list(self.board.legal_moves):
			if move.to_square == square:
				moves.append(move)
		if len(moves) < 1:
			return False

		best_move = [-1,-1]
		for move in moves:
			state = self.enc.encode(self.board, self.board.turn)
			state = np.array([state])
			state = state.astype(float)
			state = tf.convert_to_tensor(state)
			state = tf.expand_dims(state, 0)
			action_probs, critic_value  = self.model(state)
			if (critic_value > best_move[1]):
				best_move = [move, critic_value]

		return best_move[0]

	def run(self):
		max_steps = 200
		self.episode_reward = 0
		with tf.GradientTape() as tape:
			for step in range(1, max_steps):
				state = self.enc.encode(self.board, self.board.turn)
				state = np.array([state])
				state = state.astype(float)
				state = tf.convert_to_tensor(state)
				state = tf.expand_dims(state, 0)
				# Predict action probabilities and estimated future rewards
				# from environment state
				action_probs, critic_value  = self.model(state)
				#if step:
					#print (step, " reward: ", self.reward_clipping(evaluate_board(self.board, self.player_color)), "critic value:", float(critic_value))
				#print(action_probs)
				#action_probs, critic_value = model(state)
				self.critic_value_history.append(critic_value[0, 0])

				# Sample action from action probability distribution
				num_actions = 64
				if np.isnan(action_probs).all():
					self.reset()
					return
				action = np.random.choice(num_actions, p=np.squeeze(action_probs))
				#self.action_probs_history.append(tf.math.log(action_probs[0, action]))
				#square_name = chess.SQUARE_NAMES[action]
				#square = chess.parse_square(square_name)
				move = self.get_best_move(chess.SQUARES[action])
				wrong = 0
				max_wrongs = 3
				action_prob = action
				while (not move):
					action = np.random.choice(num_actions, p=np.squeeze(action_probs))
					action_prob = action
					move = self.get_best_move(chess.SQUARES[action])
					wrong += 1
					if (wrong > max_wrongs):
						move = random.choice(list(self.board.legal_moves))
						action = chess.SQUARES[move.to_square]
						action_probe = action
				self.action_probs_history.append(tf.math.log(action_probs[0, action_prob]))
				# Apply the sampled action in our environment
				#print(action)
				#print(action_probs)
				reward, done = self.env_step(move, self.board)
				
				if step % 10 == 1:
					print (step, " reward: ", self.reward_clipping(evaluate_board(self.board, self.player_color)), "critic value:", float(critic_value))
				
				#print(reward)
				self.rewards_history.append(reward)
				self.episode_reward += reward
				if done:
					break
				op_move = self.opponent.move()
				self.board.push(op_move)
				if self.board.is_game_over():
					break

			gamma = 0.99
			eps = np.finfo(np.float32).eps.item() 
			self.running_reward = 0.05 * self.episode_reward + (1 - 0.05) * self.running_reward
			print("running_reward:", self.running_reward)

			returns = []
			discounted_sum = 0
			for r in self.rewards_history[::-1]:
				discounted_sum = r + gamma * discounted_sum
				#print(discounted_sum)
				returns.insert(0, discounted_sum)
			#print(returns)
			returns = np.array(returns)
			returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
			returns = returns.tolist()
			#print(returns)
			history = zip(self.action_probs_history, self.critic_value_history, returns)
			actor_losses = []
			critic_losses = []
			for log_prob, value, ret in history:

				actor_losses.append(-log_prob * diff)  # actor loss

				critic_losses.append(
				self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
				)

			loss_value = sum(actor_losses) + sum(critic_losses)
			print(sum(critic_losses))
			print(sum(actor_losses))
			grads = tape.gradient(loss_value, self.model.trainable_variables)
			#print(grads)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
			print(loss_value)
			self.action_probs_history.clear()
			self.critic_value_history.clear()
			self.rewards_history.clear()
			return self.board.result()

	def reward_clipping(self, reward):
		return reward
		if reward > 3000:
			reward = 1.0
		elif reward > 500:
			reward = reward / 1000
			if reward > 1.0:
				reward = 0.9
		elif reward < 0:
			reward = reward / 1000
			if reward < -1.0:
				reward = -0.9
		else:
			reward = -0.1
		return reward

	def env_step(self, action, board):
		board.push(action)
		reward = evaluate_board(self.board, self.player_color)
		
		reward = self.reward_clipping(reward)

		return reward, self.board.is_game_over()

	def update(self):
		#self.reset()
		pass

	def save_model(self, model_filepath):
		self.model.save_weights(model_filepath)

	def load_model(self, model_filepath, exitOnFail=False):
		if os.path.exists(model_filepath):
			self.model.load_weights(model_filepath)
		else:
			print('Cant load model')
			if exitOnFail:
				exit()

trainable_bot = model_wrapper()
trainable_bot.load_model(model_filepath)


def play_game():
	board = MyBoard()
	trainable_bot.new_game(board)
	trainable_bot.player_color = chess.WHITE
	return trainable_bot.run()


games_count = 50
game_stats = {chess.WHITE: 0, chess.BLACK: 0}
for game_i in range(games_count):
	start = time.time()
	game_result = play_game()
	print(game_result)
	trainable_bot.update()
	if game_result == "1-0":
		game_stats[chess.WHITE] = game_stats[chess.WHITE] + 1
	elif game_result == "0-1":
		game_stats[chess.BLACK] = game_stats[chess.BLACK] + 1
	if game_i % 10 == 1:
		trainable_bot.save_model(model_filepath)
	print("elapsed:" + str(time.time()-start))
trainable_bot.save_model(model_filepath)

print(game_stats)
