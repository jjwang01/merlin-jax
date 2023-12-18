import keras
import os

os.environ["KERAS_BACKEND"] = "jax"


class ResNetBlock(keras.Layer):
    def __init__(self, num_channels, bottleneck_size, stride, kernel_size=1, activation='relu', **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        # no batch normalization used
        self.num_channels = num_channels 
        self.bottleneck_size = bottleneck_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation

    def call(self, x):
        fx = keras.layers.Conv2D(self.bottleneck_size, self.kernel_size, activation=self.activation)(x)
        fx = keras.layers.Conv2D(self.bottleneck_size, self.kernel_size, self.stride, activation=self.activation)(fx)
        fx = keras.layers.Conv2D(self.num_channels, self.kernel_size)(fx)

        x = keras.layers.Conv2D(self.num_channels, self.kernel_size, self.stride) 

        res = keras.layers.Add()([fx, x])
        # no final activation on the outputs, does that mean no ReLU here?
        return keras.layers.ReLU()(res) 
    
    def compute_output_shape(self, input_shape):
        # assuming (B, H, W, C)
        batch_size, height, width, _ = input_shape
        return (batch_size, height // self.stride, width // self.stride, self.num_channels)


class ImageEncoder(keras.Layer):
    def __init__(self, output_size=500, num_channels=64, bottleneck_size=32, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)

        self.output_size = output_size 
        self.num_channels = num_channels
        self.bottleneck_size = bottleneck_size
        self.strides = (2, 1, 2, 1, 2, 1)
    
    def build(self, input_shape):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=input_shape))
        for stride in self.strides:
            self.model.add(ResNetBlock(self.num_channels, self.bottleneck_size, stride))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.output_size), activation='tanh')
    
    def call(self, inputs):
        return self.model()(inputs)
        

class VelocityEncoder(keras.Layer):
    def __init__(self, output_size=10, scale=1000, **kwargs):
        super(VelocityEncoder, self).__init__(**kwargs)

        # b/c DM Lab environments have large scale velocities
        self.output_size = output_size 
        self.scale = scale
    
    def build(self, input_shape):
        # input_shape will usually be (B, 6)
        # v_x, v_y, v_z, w_x, w_y, w_z (3 dims for translational velocity, 3 dims for angular velocity)
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                # assume that a ReLU exists
                keras.layers.Dense(self.output_size)
            ]
        )

    def call(self, inputs):
        return self.model()(inputs / self.scale)


# Action is left unchanged as a batch of one-hot vectors
# Reward is left unchanged as a scalar magnitude


class TextEncoder(keras.Layer):
    def __init__(self, lstm_width=100, vocab_size=1000, embed_size=50, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)

        self.lstm_width = lstm_width
        self.vocab_size = vocab_size
        self.embed_size = embed_size
    
    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Embedding(self.vocab_size, self.embed_size),
                keras.layers.LSTM(self.lstm_width)
            ]
        )
    
    def call(self, inputs):
        # assuming that the text has already been tokenized into one-hots of vocab_size
        # just return the terminal hidden state
        return self.model(inputs)