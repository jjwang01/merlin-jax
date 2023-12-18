import keras
import os
import jax
import jax.numpy as jnp
import optax

os.environ["KERAS_BACKEND"] = "jax"

class MemoryBasedPredictor():
    def __init__(self, lstm_width_1, lstm_width_2, latent_size, num_mems_accessed, activation='tanh', **kwargs):
        super(MemoryBasedPredictor, self).__init__(**kwargs)

        self.lstm_width_1 = lstm_width_1 
        self.lstm_width_2 = lstm_width_2
        self.latent_size = latent_size
        self.num_mems_accessed = num_mems_accessed
        self.activation = activation

        self.memory = jnp.zeros(self.num_mems, 2 * self.latent_size)
        self.fx_1 = None
        self.fx_2 = None
        self.read_key_vectors = None
        self.read_scalars = None

        self.write_weight = jnp.zeros(self.num_mems)
        self.retro_weight = jnp.zeros(self.num_mems)
    
    def call(self, x):
        # input should be a concatenation of latent rep, action one-hot vector, 
        # and K^r (num_mems_accessed) vectors read from memory at prev time step
        
        # this is a deep LSTM that consists of two LSTM layers in parallel
        self.lstm_1 = keras.layers.LSTM(self.lstm_width_1, activation=self.activation)
        self.lstm_2 = keras.layers.LSTM(self.lstm_width_2, activation=self.activation)
        self.fx_1 = self.lstm_1()(x)
        self.fx_2 = self.lstm_2()(x)
        fx_lstm = keras.layers.Concatenate()([self.fx_1, self.fx_2])

        fx_project = keras.layers.Dense(self.num_mems_accessed * (2 * self.latent_size + 1))(fx_lstm)
        fx_reshaped = keras.layers.Reshape(shape=(self.num_mems_accessed, 2 * self.latent_size + 1))(fx_project)

        self.read_key_vectors = jnp.take(fx_reshaped, [i for i in range(2 * self.latent_size)], axis=1) 
        self.read_scalars = keras.activations.softplus(jnp.take(fx_reshaped, [-1], axis=1))

    def read(self):
        res = []
        res.append([self.fx_1, self.fx_2])
        for i in range(self.read_key_vectors.shape[0]):
            sims = [] 
            for j in range(self.memory.shape[0]):
                sims.append(optax.cosine_similarity(self.read_key_vectors[i], self.memory[j]))
            normalized_sims = jax.nn.softmax(jnp.array(self.read_scalars[i] * sims))
            memory_readout = jnp.matmul(self.memory, normalized_sims)
            res.append(memory_readout)
        return jnp.array(res) 

    def write(self, time_step):
        self.write_weight = 
        self.memory = self.memory + self.write_weight * 

    def prior(self):
    
    def posterior(self):

    def generate_state_variable(self):

class PolicyPredictor():