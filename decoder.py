import keras
import os

os.environ["KERAS_BACKEND"] = "jax"


class ResNetBlockTranspose(keras.Layer):
    def __init__(self, num_channels, bottleneck_size, stride, kernel_size=1, activation='relu', **kwargs):
        super(ResNetBlockTranspose, self).__init__(**kwargs)

        # no batch normalization used
        self.num_channels = num_channels 
        self.bottleneck_size = bottleneck_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation

    def call(self, x):
        fx = keras.layers.Conv2DTranspose(self.bottleneck_size, self.kernel_size, activation=self.activation)(x)
        fx = keras.layers.Conv2DTranspose(self.bottleneck_size, self.kernel_size, self.stride, activation=self.activation)(fx)
        fx = keras.layers.Conv2DTranspose(self.num_channels, self.kernel_size)(fx)

        x = keras.layers.Conv2DTranspose(self.num_channels, self.kernel_size, self.stride) 

        res = keras.layers.Add()([fx, x])
        # no final activation on the outputs, does that mean no ReLU here?
        return keras.layers.ReLU()(res) 
    
    def compute_output_shape(self, input_shape):
        # assuming (B, H, W, C)
        batch_size, height, width, _ = input_shape
        return (batch_size, height // self.stride, width // self.stride, self.num_channels)


class ImageDecoder(keras.Layer):
    # output_size should be formatted to the likelihood loss used for image reconstruction 
    def __init__(self, output_size=500, num_channels=64, bottleneck_size=32, activation='tanh', **kwargs):
        super(ImageDecoder, self).__init__(**kwargs)

        self.output_size = output_size 
        self.num_channels = num_channels
        self.bottleneck_size = bottleneck_size
        self.activation = activation
        self.strides = (2, 1, 2, 1, 2, 1)
    
    def build(self, input_shape):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=input_shape))
        for stride in reversed(self.strides):
            self.model.add(ResNetBlockTranspose(self.num_channels, self.bottleneck_size, stride))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.output_size), activation=self.activation)
    
    def call(self, inputs):
        return self.model()(inputs)

class StateValueFunction(keras.Layer):
    """
    First network in ReturnPredictionDecoder.

    Input is the concatenation of the latent variable (z_t) and policy pi's multinomial logits
    Gradients should NOT flow through this network!
    """
    def __init__(self, output_size, hidden_layer_size, activation, **kwargs):
        super(StateValueFunction, self).__init__(**kwargs)

        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
    
    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(self.hidden_layer_size, activation=self.activation),
                keras.layers.Dense(self.output_size),
            ]
        )
    
    def call(self, inputs):
        return self.model()(inputs)


class StateActionAdvantageFunction(keras.Layer):
    """
    Second network in ReturnPredictionDecoder.

    Input is the concatenation of latent variable (z_t) and one hot action vector (a_t).
    Gradients should flow through this network.
    """
    def __init__(self, output_size, hidden_layer_size, activation, **kwargs):
        super(StateActionAdvantageFunction, self).__init__(**kwargs)

        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
    
    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(self.hidden_layer_size, activation=self.activation),
                keras.layers.Dense(self.hidden_layer_size, activation=self.activation),
                keras.layers.Dense(self.output_size),
            ]
        )
    
    def call(self, inputs):
        return self.model()(inputs)


class ReturnPredictionDecoder(keras.Layer):
    def __init__(self, output_size=1, sv_hidden_layer_size=200, saa_hidden_layer_size=50, activation='tanh', **kwargs):
        # input is the concatenation of the latent variable (z_t), w/ policy pi's multinomial logits
        super(ReturnPredictionDecoder, self).__init__(**kwargs)

        self.sv = StateValueFunction(output_size, sv_hidden_layer_size, activation)
        self.saa = StateActionAdvantageFunction(output_size, saa_hidden_layer_size, activation)

    def call(self, x):
        # add stop gradient to state-value function
        fx_sv = self.sv()(x)
        fx_sv_stop = keras.ops.stop_gradient(fx_sv)
        fx_saa = self.saa()(x)
        res = keras.layers.Add()([fx_sv_stop, fx_saa])
    

class TextDecoder(keras.Layer):
    def __init__(self, lstm_width=100, vocab_size=1000, activation='softmax', **kwargs):
        super(TextDecoder, self).__init__(**kwargs)
        
        self.lstm_width = lstm_width
        self.vocab_size = vocab_size
        self.activation = activation

    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.LSTM(self.vocab_size, activation=self.activation)
            ]
        ) 

    def call(self, inputs):
        return self.model()(inputs)


# reward, velocity, action decoders are all latent variable (z_t) to their scalar/vector representations
class RewardDecoder(keras.Layer):
    def __init__(self, output_size=1, activation='relu', **kwargs):
        self.output_size = output_size
        # not sure if there's an activation here
        self.activation = activation

    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(self.output_size, activation=self.activation)
            ]
        )

    def call(self, inputs):
        return self.model()(inputs)
    


class VelocityDecoder(keras.Layer):
    def __init__(self, output_size=6, activation='relu', **kwargs):
        self.output_size = output_size
        # not sure if there's an activation here
        self.activation = activation

    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(self.output_size, activation=self.activation)
            ]
        )

    def call(self, inputs):
        return self.model()(inputs)
    


class ActionDecoder(keras.Layer):
    def __init__(self, output_size, activation='relu', **kwargs):
        # needs to be provided as an arg / action cardinality
        self.output_size = output_size
        # not sure if there's an activation here
        self.activation = activation

    def build(self, input_shape):
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.Dense(self.output_size, activation=self.activation)
            ]
        )

    def call(self, inputs):
        return self.model()(inputs)
    





