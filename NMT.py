import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM, Dense, Activation, RepeatVector, Layer
from tensorflow.keras.models import Model
from utils import softmax 

class AttentionLayer(Layer):
    def __init__(self, Tx, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
        self.repeat_decoder_state = RepeatVector(Tx) # repeat s_a to 
        self.concat_states = Concatenate(axis=-1)
        self.attention_hidden_layer = Dense(10, activation="tanh")
        self.attention_scoring_layer = Dense(1, activation="relu")
        self.attention_weights_softmax = Activation('softmax', name='attention_weights') # Or your custom softmax
        self.context_vector_dot = Dot(axes=1) 

    def call(self, a, s_prev):
        # Repeat the decoder's state Tx times to match the encoder's sequence length
        s_prev_repeat = self.repeat_decoder_state(s_prev) # Shape: (batch_size, Tx, n_s)

        # Glue the encoder states and repeated decoder state together
        # Shape: (batch_size, Tx, 2*n_a + n_s)
        concat = self.concat_states([a, s_prev_repeat])

        e = self.attention_hidden_layer(concat) 
        energies = self.attention_scoring_layer(e)# Shape: (batch_size, Tx, 10)
        alphas = self.attention_weights_softmax(energies) # Shape: (batch_size, Tx, 1)
        context = self.context_vector_dot([alphas, a])# Shape: (batch_size, 2 * n_a)
        return context
    
class NMTModelBuilder:
    """
    A Factory class to build the NMT Functional Model.
    """
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        self.Tx = Tx
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size

        #  Instantiate custom Attention Layer
        self.attention_layer = AttentionLayer(self.Tx)
        
        # Instantiate the shared Decoder and Output layers
        self.post_activation_LSTM_cell = LSTM(self.n_s, return_state=True) 
        self.output_layer = Dense(self.machine_vocab_size, activation=softmax) 

    def build_model(self):
        # Define Inputs
        X = Input(shape=(self.Tx, self.human_vocab_size), name="Human_Date_Input")
        s0 = Input(shape=(self.n_s,), name="Initial_Hidden_State")
        c0 = Input(shape=(self.n_s,), name="Initial_Cell_State")

        s = s0
        c = c0
        outputs = [] 

        # Pre-attention Bi-LSTM (The Encoder)
        a = Bidirectional(LSTM(self.n_a, return_sequences=True))(X)
        
        # Iterate for Ty steps (The Decoder)
        for t in range(self.Ty):
            context = self.attention_layer(a, s)
            
            # Apply the post-attention LSTM cell
            s, _, c = self.post_activation_LSTM_cell(context, initial_state=[s, c])
            
            # Apply Dense layer
            y = self.output_layer(s)
            outputs.append(y)

        model = Model(inputs=[X, s0, c0], outputs=outputs)
        return model

