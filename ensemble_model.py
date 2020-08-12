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
WEIGHT_DECAY = 0.0005
WEIGHT_PATH = ['./ensemble_model_weight/epoch_9_train_acc_0.739.h5', './ensemble_model_weight/epoch_9_train_acc_0.743.h5', './ensemble_model_weight/epoch_9_train_acc_0.741.h5']


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

backbone_1 = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=[IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3])
loss_model_1 = AdaCos(NUM_TRAIN_LABEL)

backbone_2 = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=[IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3])
loss_model_2 = AdaCos(NUM_TRAIN_LABEL, regularizer=regularizers.l2(WEIGHT_DECAY))

backbone_3 = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=[IMAGE_MAX_SIZE, IMAGE_MAX_SIZE, 3])
loss_model_3 = AdaCos(NUM_TRAIN_LABEL, regularizer=regularizers.l2(WEIGHT_DECAY))

for layer in backbone_2.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(WEIGHT_DECAY))

for layer in backbone_3.layers:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(WEIGHT_DECAY))


model_1 = tf.keras.Sequential([
    backbone_1,
    Generalized_mean_pooling2D(),
    Dense(EMBEDDING_SIZE, name='fc'),
    BatchNormalization(name='batchnorm'),
    loss_model_1
])
model_1.load_weights(WEIGHT_PATH[0])

model_2 = tf.keras.Sequential([
    backbone_2,
    Generalized_mean_pooling2D(),
    Dense(EMBEDDING_SIZE, name='fc'),
    BatchNormalization(name='batchnorm'),
    loss_model_2
])
model_2.load_weights(WEIGHT_PATH[1])

model_3 = tf.keras.Sequential([
    backbone_3,
    Generalized_mean_pooling2D(),
    Dense(EMBEDDING_SIZE, name='fc'),
    BatchNormalization(name='batchnorm'),
    loss_model_3
])
model_3.load_weights(WEIGHT_PATH[2])


feature_extractor_1 = Model(inputs=model_1.inputs, outputs=model_1.get_layer('batchnorm').output)
feature_extractor_2 = Model(inputs=model_2.inputs, outputs=model_2.get_layer('batchnorm').output)
feature_extractor_3 = Model(inputs=model_3.inputs, outputs=model_3.get_layer('batchnorm').output)

feature_extractor_1.summary()
feature_extractor_2.summary()
feature_extractor_3.summary()

models = []
models.append(feature_extractor_1)
models.append(feature_extractor_2)
models.append(feature_extractor_3)

class MyModel(tf.keras.Model):
    def __init__(self, models):
        super(MyModel, self).__init__()
        self.models = models
        for model in self.models:
            model.trainable = False
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
    ])
    def call(self, input_image):
        output_tensors = {}
        
        # resizing
        im = tf.image.resize_with_pad(input_image, 441, 441)
        
        # preprocessing
        im = tf.cast(im, tf.float32)
        im = tf.math.divide(tf.subtract(im, 127.5), 127.5)

        features = []
        for model in self.models:
            extracted_features = model(tf.convert_to_tensor([im]))
            extracted_features = tf.math.l2_normalize(extracted_features[0])
            features.append(extracted_features)

        ensemble_feature = tf.keras.backend.concatenate(features, axis=-1)
        print(ensemble_feature)

        output_tensors['global_descriptor'] = tf.identity(ensemble_feature, name='global_descriptor')
        return output_tensors

m = MyModel(models)

served_function = m.call
tf.saved_model.save(m, export_dir="./457_ensemble_model", signatures={'serving_default': served_function})

# from zipfile import ZipFile

# with ZipFile('submission.zip','w') as zip:           
#     zip.write('./my_model/saved_model.pb', arcname='saved_model.pb') 
#     zip.write('./my_model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001')
#     zip.write('./my_model/variables/variables.index', arcname='variables/variables.index')