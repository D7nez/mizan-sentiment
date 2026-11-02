import tensorflow as tf
@tf.keras.utils.register_keras_serializable(package='mizan')
class SelfAttention(tf.keras.layers.Layer):
    """Legacy attention preserves original inference semantics (padding unmasked)."""
    def build(self,input_shape):
        self.W=self.add_weight(shape=(input_shape[-1],1),initializer='glorot_uniform',trainable=True)
        self.b=self.add_weight(shape=(input_shape[1],1),initializer='zeros',trainable=True)
        super().build(input_shape)
    def call(self,x):
        scores=tf.tanh(tf.matmul(x,self.W)+self.b)
        return tf.reduce_sum(x*tf.nn.softmax(scores,axis=1),axis=1)

def build_model(vocab=60000,length=150):
    inputs=tf.keras.Input(shape=(length,))
    x=tf.keras.layers.Embedding(vocab,128)(inputs)
    x=tf.keras.layers.SpatialDropout1D(.3)(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True,recurrent_dropout=.4,kernel_regularizer=tf.keras.regularizers.l2(.01)))(x)
    x=SelfAttention()(x);x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(.01))(x)
    x=tf.keras.layers.Dropout(.4)(x)
    return tf.keras.Model(inputs,tf.keras.layers.Dense(3,activation='softmax')(x))
