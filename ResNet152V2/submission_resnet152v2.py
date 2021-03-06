import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers, Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

IMAGE_MAX_SIZE = 441
EMBEDDING_SIZE = 512
NUM_TRAIN_LABEL = 81313
WEIGHT_PATH = "./ResNet152V2/LR_0.00001_weight_decay_0.0005_fine_tune/6_2_output_resnet152v2/epoch_9_train_acc_0.756.h5"


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

        # theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

        # B_avg = tf.where(y_true < 1, tf.exp(self.s*logits), tf.zeros_like(logits))
        # B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
        # theta_class = theta[y_true == 1]
        # theta_med = tfp.stats.percentile(theta_class, q=50)

        # denominator = tf.cos(tf.minimum(math.pi / 4.0, theta_med))
        # numerator = tf.math.log(B_avg)
        # self.s.assign(numerator / denominator)

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

backbone = tf.keras.applications.ResNet152V2(include_top=False, weights=None, input_shape=[IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3], classifier_activation=None)

loss_model = AdaCos(NUM_TRAIN_LABEL)

entire_model = tf.keras.Sequential([
    backbone,
    Generalized_mean_pooling2D(),
    Dense(EMBEDDING_SIZE, name='fc'),
    BatchNormalization(name='batchnorm'),
    loss_model
])
entire_model.load_weights(WEIGHT_PATH)

feature_extractor = Model(inputs=entire_model.inputs, outputs=entire_model.get_layer('batchnorm').output)
entire_model.summary()
feature_extractor.summary()

class MyModel(tf.keras.Model):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model
        self.model.trainable = False
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')])
    def call(self, input_image):
        output_tensors = {}
        
        # resizing
        im = tf.image.resize_with_pad(input_image, 441, 441)
        
        # preprocessing
        im = tf.cast(im, tf.float32)
        im = tf.math.divide(tf.subtract(im, 127.5), 127.5)
        
        extracted_features = self.model(tf.convert_to_tensor([im]))
        features = tf.math.l2_normalize(extracted_features[0])
        output_tensors['global_descriptor'] = tf.identity(features, name='global_descriptor')
        return output_tensors

m = MyModel(feature_extractor) #creating our model instance

served_function = m.call
tf.saved_model.save(m, export_dir="./6_2_resnet152v2_finetune", signatures={'serving_default': served_function})

# from zipfile import ZipFile

# with ZipFile('submission.zip','w') as zip:           
#     zip.write('./my_model/saved_model.pb', arcname='saved_model.pb') 
#     zip.write('./my_model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001')
#     zip.write('./my_model/variables/variables.index', arcname='variables/variables.index')