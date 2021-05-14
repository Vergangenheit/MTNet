from typing import TypeVar, List


class TaxiNYConfig:
    T: int = 12  # timestep
    W: int = 6  # convolution window size (convolution filter height)
    n: int = 6  # the number of the long-term memory series
    highway_window: int = 12  # the window size of ar model

    D: int = 100  # input's variable dimension (convolution filter width)
    K: int = 100  # output's variable dimension
    horizon: int = 1  # the horizon of predicted value

    en_conv_hidden_size: int = 32
    en_rnn_hidden_sizes: List = [16, 32]  # last size is equal to en_conv_hidden_size

    input_keep_prob: float = 0.8
    output_keep_prob: float = 1.0

    lr: float = 0.001

    batch_size: int = 100


class BJpmConfig:

    def __init__(self):
        self.T = 8  # timestep
        self.W = 3  # convolution window size (convolution filter height)`
        self.n = 7  # the number of the long-term memory series
        self.highway_window = 8  # the window size of ar model

        self.D = 8  # input's variable dimension (convolution filter width)
        self.K = 1  # output's variable dimension

        self.horizon = 6  # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.001
        self.batch_size = 100


class SolarEnergyConfig:

    def __init__(self):
        self.T = 12  # timestep
        self.W = 6  # convolution window size (convolution filter height)`
        self.n = 7  # the number of the long-term memory series
        self.highway_window = 6  # the window size of ar model

        self.D = 137  # input's variable dimension (convolution filter width)
        self.K = 137  # output's variable dimension

        self.horizon = 6  # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [20, 32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.003
        self.batch_size = 100


class BikeNYCConfig:

    def __init__(self):
        self.T = 24  # timestep
        self.W = 6  # convolution window size (convolution filter height)`
        self.n = 6  # the number of the long-term memory series
        self.highway_window = 12  # the window size of ar model

        self.D = 256  # input's variable dimension (convolution filter width)
        self.K = 256  # output's variable dimension

        self.horizon = 1  # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [32, 32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.003
        self.batch_size = 32


ConfigType = TypeVar("configuration class", BikeNYCConfig, TaxiNYConfig, SolarEnergyConfig, BJpmConfig)
