import os
import re
import math
import functools
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers, Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
SKIP_VALIDATION = True
BATCH_SIZE = 8 * 8
EPOCHS = 20
IMAGE_NORM_MODE = 1  # 0: 0 ~ 1, 1: -1 ~ 1
IMAGE_MAX_SIZE = 441
EMBEDDING_SIZE = 512
WEIGHT_DECAY = 0.0005

LR_START = 0.001

TPU_IP = '10.240.1.10'

train_tfrec_dir = 'gs://landmark-train-set/tfrec/train*'
valid_tfrec_dir = "gs://landmark-train-set/tfrec/valid*"

TRAINING_FILENAMES = tf.io.gfile.glob(train_tfrec_dir)
VALIDATION_FILENAMES = tf.io.gfile.glob(valid_tfrec_dir)

if SKIP_VALIDATION:
    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAIN_LABEL = 81313
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = 0 if SKIP_VALIDATION else count_data_items(VALIDATION_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + TPU_IP)  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.MirroredStrategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

def _parse_example(example):
    name_to_features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    
    parsed_example = tf.io.parse_single_example(example, name_to_features)
    
    # Parse to get image.
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    if IMAGE_NORM_MODE == 0:
        image = tf.math.divide(image, 255.0)
    else:
        image = tf.math.divide(tf.subtract(image, 127.5), 127.5)

    # Resize image.
    image = tf.image.resize_with_pad(image, IMAGE_MAX_SIZE, IMAGE_MAX_SIZE)
    image = tf.reshape(image, [IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3]) # explicit size needed for TPU
    
    # Parse to get label.
    label = tf.cast(parsed_example['image/class/label'], tf.int32)
    label = tf.one_hot(label, NUM_TRAIN_LABEL)

    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_example, num_parallel_calls=AUTO)
    return dataset

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return tf.keras.backend.dot(tf.keras.backend.dot(rotation_matrix, shear_matrix), tf.keras.backend.dot(zoom_matrix, shift_matrix))


def transform(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_MAX_SIZE
    XDIM = DIM % 2

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_hue(image, 0.2)
    
    rot = 15. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32') 
    w_shift = 16. * tf.random.normal([1],dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.keras.backend.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = tf.keras.backend.cast(idx2,dtype='int32')
    idx2 = tf.keras.backend.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3]),label

def get_dataset(file_name, augmentation=True, validation=False):
    dataset = load_dataset(file_name)
    
    if augmentation: 
        dataset = dataset.map(transform, num_parallel_calls=AUTO)
    
    if validation:
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    else:
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


class Generalized_mean_pooling2D(Layer):
    def __init__(self, p=3, epsilon=1e-6, **kwargs):
        super(Generalized_mean_pooling2D, self).__init__(**kwargs)

        self.init_p = p
        self.epsilon = epsilon

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')

        self.build_shape = input_shape

        self.p = self.add_weight(
              name='p',
              shape=[1,],
              initializer=tf.keras.initializers.Constant(value=self.init_p),
              regularizer=None,
              trainable=True,
              dtype=tf.float32
              )

        self.built=True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')

        return (tf.reduce_mean(tf.abs(inputs**self.p), axis=[1,2], keepdims=False) + self.epsilon)**(1.0/self.p)
    
class AdaCos(Layer):
    def __init__(self, n_classes=10, regularizer=None, **kwargs):
        super(AdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(AdaCos, self).build(input_shape)
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        self.s = tf.Variable(tf.math.sqrt(2.0)*tf.math.log(self.n_classes - 1.0), trainable=False, aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs):
        # normalize feature
        x = tf.nn.l2_normalize(inputs, axis=1, name='norm_embeddings')
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0, name='norm_loss_weights')
        # dot product
        logits = x @ W
        
        return logits
    
    def get_logits(self, y_true, y_pred):
        logits = y_pred

        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        theta_class = theta[y_true == 1]
        theta_med = tfp.stats.percentile(theta_class, q=50)

        self.s.assign(tf.math.log(self.n_classes - 1.0) / tf.cos(tf.minimum(math.pi / 4.0, theta_med)))

        logits = self.s * logits
        out = tf.nn.softmax(logits)

        return out

    def loss(self, y_true, y_pred):
        logits = self.get_logits(y_true, y_pred)
        loss = tf.keras.losses.categorical_crossentropy(y_true, logits)
        return loss

    def accuracy(self, y_true, y_pred):
        logits = self.get_logits(y_true, y_pred)
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, logits)
        return accuracy
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)


with strategy.scope():
    backbone = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=[IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3])
        
    for layer in backbone.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(WEIGHT_DECAY))

    loss_model = AdaCos(NUM_TRAIN_LABEL, regularizer=regularizers.l2(WEIGHT_DECAY))
    
    entire_model = tf.keras.Sequential([
        backbone,
        Generalized_mean_pooling2D(),
        Dense(EMBEDDING_SIZE, name='fc'),
        BatchNormalization(name='batchnorm'),
        loss_model
    ], name='Landmark_Retrieval_2020_Model_{}'.format(backbone.name))
    
entire_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = tf.keras.experimental.CosineDecay(LR_START, STEPS_PER_EPOCH * EPOCHS)),
            loss = loss_model.loss,
            metrics = [loss_model.accuracy])

entire_model.summary()

class ModelSaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        os.makedirs('./output_{}'.format(entire_model.layers[0].name), exist_ok=True)
        entire_model.save_weights('./output_{0}/epoch_{1}_train_acc_{2:.3f}.h5'.format(entire_model.layers[0].name, epoch, logs['accuracy']))

history = entire_model.fit(
    get_dataset(TRAINING_FILENAMES), 
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[ModelSaveCallback()],
    validation_data=None if SKIP_VALIDATION else get_dataset(VALIDATION_FILENAMES, validation=True)
)