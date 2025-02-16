from tensorflow.keras.layers import Input, Permute, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, Dense, Activation, Reshape
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(128, 1, 37), name='input_1', dtype='float32')

# Permute layer
permute = Permute((1, 3, 2), name='permute_1')(input_layer)

# First Conv2D block
conv2d_1 = Conv2D(32, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_1')(permute)
conv2d_2 = Conv2D(32, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_2')(conv2d_1)
batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(conv2d_2)
leaky_re_lu_1 = LeakyReLU(alpha=0.3, name='leaky_re_lu_1')(batch_normalization_1)
max_pooling2d_1 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same', name='max_pooling2d_1')(leaky_re_lu_1)
dropout_1 = Dropout(0.5, name='dropout_1')(max_pooling2d_1)

# Second Conv2D block
conv2d_3 = Conv2D(32, (1, 37), activation='linear', padding='valid', use_bias=False, name='conv2d_3')(dropout_1)
conv2d_4 = Conv2D(64, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_4')(conv2d_3)
batch_normalization_2 = BatchNormalization(name='batch_normalization_2')(conv2d_4)
leaky_re_lu_2 = LeakyReLU(alpha=0.3, name='leaky_re_lu_2')(batch_normalization_2)
dropout_2 = Dropout(0.5, name='dropout_2')(leaky_re_lu_2)

# Third Conv2D block
conv2d_5 = Conv2D(64, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_5')(dropout_2)
batch_normalization_3 = BatchNormalization(name='batch_normalization_3')(conv2d_5)
leaky_re_lu_3 = LeakyReLU(alpha=0.3, name='leaky_re_lu_3')(batch_normalization_3)
max_pooling2d_2 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same', name='max_pooling2d_2')(leaky_re_lu_3)
dropout_3 = Dropout(0.5, name='dropout_3')(max_pooling2d_2)

# Fourth Conv2D block
conv2d_6 = Conv2D(96, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_6')(dropout_3)
batch_normalization_4 = BatchNormalization(name='batch_normalization_4')(conv2d_6)
leaky_re_lu_4 = LeakyReLU(alpha=0.3, name='leaky_re_lu_4')(batch_normalization_4)
dropout_4 = Dropout(0.5, name='dropout_4')(leaky_re_lu_4)

# Fifth Conv2D block
conv2d_7 = Conv2D(96, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_7')(dropout_4)
batch_normalization_5 = BatchNormalization(name='batch_normalization_5')(conv2d_7)
leaky_re_lu_5 = LeakyReLU(alpha=0.3, name='leaky_re_lu_5')(batch_normalization_5)
max_pooling2d_3 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same', name='max_pooling2d_3')(leaky_re_lu_5)
dropout_5 = Dropout(0.5, name='dropout_5')(max_pooling2d_3)

# Sixth Conv2D block
conv2d_8 = Conv2D(128, (8, 1), activation='linear', padding='same', use_bias=False, name='conv2d_8')(dropout_5)
batch_normalization_6 = BatchNormalization(name='batch_normalization_6')(conv2d_8)
leaky_re_lu_6 = LeakyReLU(alpha=0.3, name='leaky_re_lu_6')(batch_normalization_6)
dropout_6 = Dropout(0.5, name='dropout_6')(leaky_re_lu_6)

# Reshape and final dense and activation layers
reshape_1 = Reshape((256,), name='reshape_1')(dropout_6)
dense_1 = Dense(1, activation='linear', use_bias=True, name='dense_1')(reshape_1)
activation_1 = Activation('sigmoid', name='activation_1')(dense_1)

# Create model
model = Model(inputs=input_layer, outputs=activation_1, name='custom_model')

model.summary()
