import tensorflow as tf


class AC_model(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.input1 = tf.keras.layers.Conv2D(192, kernel_size=(5, 5), padding='same', activation='relu')
		self.dropout1 = tf.keras.layers.Dropout(rate=0.6)
		self.innerConv = []
		for i in range(2,8):
			self.innerConv.append(tf.keras.layers.Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu'))
			#self.innerConv.append(tf.keras.layers.Dropout(rate=0.6))
		self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(512, activation='relu')
		self.dropout2 = tf.keras.layers.Dropout(rate=0.6)
		self.actor = tf.keras.layers.Dense(64, activation='softmax')
		self.critic_in1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')
		self.flatten2 = tf.keras.layers.Flatten()
		self.dense2 = tf.keras.layers.Dense(512, activation='relu')
		self.critic = tf.keras.layers.Dense(1, activation='linear')


	def call(self, inputs, training=False):
		training = False
		x = self.input1(inputs)
		if training:
			x = self.dropout1(x, training=training)
		for i in range(len(self.innerConv)):
			x = self.innerConv[i](x)
			if training:
				x = self.innerConv[i+1](x, training=training)
		x = self.conv2(x)
		x_actor = self.flatten(x)
		if training:
			x_actor = self.dropout2(x_actor, training=training)
		x_actor = self.dense1(x_actor)

		actor_value = self.actor(x_actor)

		x = self.critic_in1(x)
		x = self.flatten2(x)
		x = self.dense2(x)
		critic_value = self.critic(x)

		return (actor_value, critic_value)
