import os

import jax
import jax.numpy as jnp
import keras
import optax

os.environ["KERAS_BACKEND"] = "jax"


class Agent:
    def __init__(
        self,
        lstm_width_1,
        lstm_width_2,
        latent_size,
        num_mems,
        num_actions,
        policy_hidden_size=200,
        activation="tanh",
        **kwargs
    ):
        super(Agent, self).__init__(**kwargs)

        # Memory Based Predictor
        self.lstm_width_1 = lstm_width_1
        self.lstm_width_2 = lstm_width_2
        self.latent_size = latent_size
        self.num_mems = num_mems
        self.activation = activation

        self.memory = jnp.zeros(self.num_mems, 2 * self.latent_size)
        self.fx_1 = None
        self.fx_2 = None
        self.read_key_vectors = None
        self.read_scalars = None

        self.retro_weight = jnp.zeros(self.num_mems)
        self.usages = jnp.zeros(self.num_mems)

        # policy
        self.num_actions = num_actions
        self.policy_hidden_size = policy_hidden_size

    def call_mbp_lstm(self, x):
        # input should be a concatenation of latent rep, action one-hot vector,
        # and K^r (num_mems) vectors read from memory at prev time step

        # this is a deep LSTM that consists of two LSTM layers in parallel
        self.lstm_1 = keras.layers.LSTM(self.lstm_width_1, activation=self.activation)
        self.lstm_2 = keras.layers.LSTM(self.lstm_width_2, activation=self.activation)
        self.fx_1 = self.lstm_1()(x)
        self.fx_2 = self.lstm_2()(x)
        fx_lstm = keras.layers.Concatenate()([self.fx_1, self.fx_2])

        fx_project = keras.layers.Dense(
            self.num_mems * (2 * self.latent_size + 1), activation=self.activation
        )(fx_lstm)
        fx_reshaped = keras.layers.Reshape(
            shape=(self.num_mems, 2 * self.latent_size + 1)
        )(fx_project)

        self.read_key_vectors = jnp.take(
            fx_reshaped, [i for i in range(2 * self.latent_size)], axis=1
        )
        self.read_scalars = keras.activations.softplus(
            jnp.take(fx_reshaped, [-1], axis=1)
        )

    def read(self):
        res = jnp.concatenate((self.fx_1, self.fx_2))
        for i in range(self.read_key_vectors.shape[0]):
            sims = []
            for j in range(self.num_mems):
                sims.append(
                    optax.cosine_similarity(self.read_key_vectors[i], self.memory[j])
                )
            normalized_sims = jax.nn.softmax(jnp.array(self.read_scalars[i] * sims))
            # update usages for this read key
            self.usages = self.usages + normalized_sims
            memory_readout = jnp.matmul(self.memory, normalized_sims)
            res = res.concatenate((res, memory_readout))
        return jnp.array(res)

    def write(self, z_t, discount):
        # choose the row with the smallest usage
        # if tie, use the one with the smallest index
        min_idx = jnp.argmin(self.usages)[0]
        write_weight = jnp.array([1 if min_idx else 0 for i in range(self.num_mems)])

        self.memory = (
            self.memory
            + jnp.matmul(
                write_weight,
                jnp.transpose(jnp.concatenate((z_t, jnp.zeros(self.latent_size)))),
            )
            + jnp.matmul(
                self.retro_weight,
                jnp.transpose(jnp.concatenate((jnp.zeros(self.latent_size), z_t))),
            )
        )
        self.retro_weight = discount * self.retro_weight + (1 - discount) * write_weight

    # these two functions produce a 2 * |z| length output representing a diagonal Gaussian distribution,
    # where the first |z| is the mean, and the second |z| is the log stdev
    def prior(self, h_prev, m_prev):
        res = jnp.concatenate((h_prev, m_prev))
        for _ in range(2):
            res = keras.layers.Dense(2 * self.latent_size, activation=self.activation)(
                res
            )
        return keras.layers.Dense(2 * self.latent_size)(res)

    def posterior(self, e_t, h_prev, m_prev, mu_prior, log_sigma_prior):
        res = jnp.concatenate((e_t, h_prev, m_prev, mu_prior, log_sigma_prior))
        for _ in range(2):
            res = keras.layers.Dense(2 * self.latent_size, activation=self.activation)(
                res
            )
        return keras.layers.Dense(2 * self.latent_size)(res) + jnp.concatenate(
            (h_prev, m_prev)
        )

    # sample from posterior
    def generate_state_variable(self, mu_post, log_sigma_post):
        return mu_post + jnp.dot(
            jnp.exp(log_sigma_post), jax.random.normal(shape=(self.latent_size))
        )

    def act(self, z_t):
        # can read from memory in the same way as MBP
        # TODO: double check if this is right, the paper is vague w.r.t. this
        self.lstm_1 = keras.layers.LSTM(self.lstm_width_1, activation=self.activation)
        self.lstm_2 = keras.layers.LSTM(self.lstm_width_2, activation=self.activation)
        self.fx_1 = self.lstm_1()(z_t)
        self.fx_2 = self.lstm_2()(z_t)
        fx_lstm = keras.layers.Concatenate()([self.fx_1, self.fx_2])

        # 2 * |z| + 1 -> h_t, m_t
        fx_project = keras.layers.Dense(
            2 * self.latent_size, activation=self.activation
        )(fx_lstm)
        fx = jnp.concatenate((z_t, fx_project))
        mlp_output = keras.layers.Dense(
            self.policy_hidden_size, activation=self.activation
        )(fx)
        res = keras.layers.Dense(self.num_actions)(mlp_output)

        return jax.nn.softmax(res)
