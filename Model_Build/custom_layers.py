from keras import backend as K
from keras.layers import Layer, initializers, regularizers, constraints


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification" by using a context
    vector to assist the attention.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self, init='glorot_uniform',
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            (input_shape[-1], 1),
            initializer=self.kernel_initializer,
            name='{}_W'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.b = self.add_weight(
            (input_shape[1],),
            initializer='zero',
            name='{}_b'.format(self.name),
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint
        )
        self.u = self.add_weight(
            (input_shape[1],),
            initializer=self.kernel_initializer,
            name='{}_u'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        multdata = K.dot(x, self.kernel)  # (x, 40, 300) * (300, 1) => (x, 40, 1)
        multdata = K.squeeze(multdata, -1)  # (x, 40)
        multdata = multdata + self.b  # (x, 40) + (40,)

        multdata = K.tanh(multdata)  # (x, 40)

        multdata = multdata * self.u  # (x, 40) * (40, 1) => (x, 1)
        multdata = K.exp(multdata)  # (x, 1)

        # Apply mask after the exp. will be re-normalized next.
        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            multdata = mask * multdata  # (x, 40) * (x, 40, )

        # In some cases, especially in the early stages of training, the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multdata /= K.cast(K.sum(multdata, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multdata = K.expand_dims(multdata)
        weighted_input = x * multdata
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)
