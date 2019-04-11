
import  tensorflow as tf

import numpy as np
import keras
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.layers import ReLU
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from data_keras import label_acc_score,keras_data
# import keras
import glob
import math
import  matplotlib.pyplot as plt
# class BilinearUpsampling(Layer):
#     """Just a simple bilinear upsampling layer. Works only with TF.
#        Args:
#            upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
#            output_size: used instead of upsampling arg if passed!
#     """
#
#     def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
#
#         super(BilinearUpsampling, self).__init__(**kwargs)
#
#         self.data_format = K.image_data_format()
#         self.input_spec = InputSpec(ndim=4)
#         if output_size:
#             self.output_size = conv_utils.normalize_tuple(
#                 output_size, 2, 'output_size')
#             self.upsampling = None
#         else:
#             self.output_size = None
#             self.upsampling = conv_utils.normalize_tuple(
#                 upsampling, 2, 'upsampling')
#
#     def compute_output_shape(self, input_shape):
#         if self.upsampling:
#             height = self.upsampling[0] * \
#                 input_shape[1] if input_shape[1] is not None else None
#             width = self.upsampling[1] * \
#                 input_shape[2] if input_shape[2] is not None else None
#         else:
#             height = self.output_size[0]
#             width = self.output_size[1]
#         return (input_shape[0],
#                 height,
#                 width,
#                 input_shape[3])
#
#     def call(self, inputs):
#         if self.upsampling:
#             return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
#                                                        inputs.shape[2] * self.upsampling[1]),
#                                               align_corners=True)
#         else:
#             return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
#                                                        self.output_size[1]),
#                                               align_corners=True)
#
#     def get_config(self):
#         config = {'upsampling': self.upsampling,
#                   'output_size': self.output_size,
#                   'data_format': self.data_format}
#         base_config = super(BilinearUpsampling, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# def relu6(x):
#     return K.relu(x, max_value=6)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        # x = Activation(relu6, name=prefix + 'expand_relu')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    # x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)
    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


# def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(320, 240, 3), classes=1, backbone='mobilenetv2', OS=8, alpha=1.):
#     """ Instantiates the Deeplabv3+ architecture
#
#     Optionally loads weights pre-trained
#     on PASCAL VOC. This model is available for TensorFlow only,
#     and can only be used with inputs following the TensorFlow
#     data format `(width, height, channels)`.
#     # Arguments
#         weights: one of 'pascal_voc' (pre-trained on pascal voc)
#             or None (random initialization)
#         input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#             to use as image input for the model.
#         input_shape: shape of input image. format HxWxC
#             PASCAL VOC model was trained on (512,512,3) images
#         classes: number of desired classes. If classes != 21,
#             last layer is initialized randomly
#         backbone: backbone to use. one of {'xception','mobilenetv2'}
#         OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
#             Used only for xception backbone.
#         alpha: controls the width of the MobileNetV2 network. This is known as the
#             width multiplier in the MobileNetV2 paper.
#                 - If `alpha` < 1.0, proportionally decreases the number
#                     of filters in each layer.
#                 - If `alpha` > 1.0, proportionally increases the number
#                     of filters in each layer.
#                 - If `alpha` = 1, default number of filters from the paper
#                     are used at each layer.
#             Used only for mobilenetv2 backbone
#
#     # Returns
#         A Keras model instance.
#
#     # Raises
#         RuntimeError: If attempting to run this model with a
#             backend that does not support separable convolutions.
#         ValueError: in case of invalid argument for `weights` or `backbone`
#
#     """
#
#
#     if input_tensor is None:
#         img_input = Input(shape=input_shape)
#     else:
#         if not K.is_keras_tensor(input_tensor):
#             img_input = Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor
#
#
#
#     first_block_filters = _make_divisible(32 * alpha, 8)
#     x = Conv2D(first_block_filters,
#                kernel_size=3,
#                strides=(2, 2), padding='same',
#                use_bias=False, name='Conv')(img_input)
#     x = BatchNormalization(
#         epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
#     # x = Activation(relu6, name='Conv_Relu6')(x)
#     x = ReLU(6., name='Conv_Relu6')(x)
#     x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
#                             expansion=1, block_id=0, skip_connection=False)
#
#     x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
#                             expansion=6, block_id=1, skip_connection=False)
#     x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
#                             expansion=6, block_id=2, skip_connection=True)
#
#     x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
#                             expansion=6, block_id=3, skip_connection=False)
#     x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                             expansion=6, block_id=4, skip_connection=True)
#     x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                             expansion=6, block_id=5, skip_connection=True)
#
#     # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
#     x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
#                             expansion=6, block_id=6, skip_connection=False)
#     x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=7, skip_connection=True)
#     x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=8, skip_connection=True)
#     x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=9, skip_connection=True)
#
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=10, skip_connection=False)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=11, skip_connection=True)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                             expansion=6, block_id=12, skip_connection=True)
#
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
#                             expansion=6, block_id=13, skip_connection=False)
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                             expansion=6, block_id=14, skip_connection=True)
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                             expansion=6, block_id=15, skip_connection=True)
#
#     x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
#                             expansion=6, block_id=16, skip_connection=False)
#
#     # end of feature extractor
#
#     # branching for Atrous Spatial Pyramid Pooling
#
#
#
#     # Image Feature branch
#     #out_shape = int(np.ceil(input_shape[0] / OS))
#     b4 = AveragePooling2D(pool_size=(40,30))(x)
#     b4 = Conv2D(256, (1, 1), padding='same',
#                 use_bias=False, name='image_pooling')(b4)
#     b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
#     # b4 = Activation('relu')(b4)
#     b4 = ReLU(6.)(b4)
#     # b4 = BilinearUpsampling((40,30))(b4)
#     # b4=K.tf.image.resize_bilinear(b4, (40,30),align_corners=True)
#     b4=keras.layers.UpSampling2D((40,30),interpolation='bilinear')(b4)
#     # simple 1x1
#     b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
#     b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
#     # b0 = Activation('relu', name='aspp0_activation')(b0)
#     b0 = ReLU()(b0)
#
#     x = Concatenate()([b4, b0])
#
#     x = Conv2D(256, (1, 1), padding='same',
#                use_bias=False, name='concat_projection')(x)
#     x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
#     # x = Activation('relu')(x)
#     x= ReLU()(x)
#     x = Dropout(0.1)(x)
#
#     # DeepLab v.3+ decoder
#
#     x = Conv2D(classes, (1, 1), padding='same', name='custom')(x)
#     # x = BilinearUpsampling(output_size=(320,240))(x)
#     # x=K.tf.image.resize_bilinear(x, (320, 240), align_corners=True)
#     x = keras.layers.UpSampling2D((8, 8), interpolation='bilinear',name='logit_label')(x)
#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = get_source_inputs(input_tensor,name='main_input')
#     else:
#         inputs = img_input
#     x = keras.layers.Activation('sigmoid')(x)
#     # input=keras.layers.Input(shape=(320,240,3),name='main_input')
#     model = Model(inputs, x, name='deeplabv3plus')
#
#     # load weights
#
#     model.load_weights('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5', by_name=True)
#     return model
def mDeeplab(weights='pascal_voc', input_tensor=None, input_shape=(320, 240, 3), classes=1, backbone='mobilenetv2', OS=8, alpha=0.5,temp=1):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor



    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    # x = Activation(relu6, name='Conv_Relu6')(x)
    x = ReLU(6., name='Conv_Relu6')(x)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling



    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(40,30))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    # b4 = Activation('relu')(b4)
    b4 = ReLU(6.)(b4)
    # b4 = BilinearUpsampling((40,30))(b4)
    # b4=K.tf.image.resize_bilinear(b4, (40,30),align_corners=True)
    b4=keras.layers.UpSampling2D((40,30),interpolation='bilinear')(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    # b0 = Activation('relu', name='aspp0_activation')(b0)
    b0 = ReLU()(b0)

    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x= ReLU()(x)
    x = Dropout(0.1)(x)

    # DeepLab v.3+ decoder

    x = Conv2D(classes, (1, 1), padding='same', name='custom')(x)
    # x = BilinearUpsampling(output_size=(320,240))(x)
    # x=K.tf.image.resize_bilinear(x, (320, 240), align_corners=True)
    x = keras.layers.UpSampling2D((8, 8), interpolation='bilinear',name='logit_label')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor,name='main_input')
    else:
        inputs = img_input
    soft_x=keras.layers.Lambda(lambda x:(1/temp)*x)(x)
    soft_x=keras.layers.Activation('sigmoid',name='soft')(soft_x)
    x = keras.layers.Activation('sigmoid',name='stand')(x)
    # input=keras.layers.Input(shape=(320,240,3),name='main_input')
    model = Model(inputs=inputs, outputs=[soft_x,x])

    # load weights

    # model.load_weights('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5', by_name=True)
    return model

# def distll_loss(y_true,y_pred,alpha=0.1):
#     mask_t,softmask=y_true
#     soft_x,x=y_pred
#     return  keras.backend.binary_crossentropy(mask_t,x)*alpha+keras.backend.binary_crossentropy(softmask,soft_x)


def diceloss(y_true,y_pred):

    numerator=2*keras.backend.sum(y_true*y_pred)+0.0001
    denominator=keras.backend.sum(y_true**2)+keras.backend.sum(y_pred**2)+0.0001
    return 1-numerator/denominator/2
batch_size=2
temperature=5
alpha=0.1
train_data=keras_data(batch_size=batch_size,temp=temperature)
val_data=keras_data(image_set='test',batch_size=batch_size)
optim=keras.optimizers.Adam()
steps = math.ceil(len(glob.glob('data/mask/'+ '*.png')) / batch_size)
# model=Deeplabv3()
# model.compile(optimizer=optim,loss='binary_crossentropy')
# model.fit_generator(train_data,steps_per_epoch=steps,epochs=20,use_multiprocessing=True, verbose=2,workers=4)
# keras.models.save_model(model, 'dddtrue_weight_dsf.h5',include_optimizer=False)




model=mDeeplab(temp=temperature)
model.compile(optimizer=optim,loss={'soft':'binary_crossentropy','stand':'binary_crossentropy'},loss_weights={'soft':1-alpha,'stand':alpha})
model.fit_generator(train_data,steps_per_epoch=steps,epochs=20,use_multiprocessing=True, verbose=2,workers=4)
keras.models.save_model(model,'distill.h5',include_optimizer=False)

# model=keras.models.load_model('distill.h5')
# tmp_model=Model(inputs=model.input,outputs=[model.get_layer('logit_label').output,model.get_layer('lambda_1').output])
# a,b=next(iter(train_data))
# q,w=model.predict(a)



# with keras.utils.generic_utils.CustomObjectScope({'BilinearUpsampling':BilinearUpsampling}):
# converter=tf.lite.TFLiteConverter.from_keras_model_file('dddtrue_weight_dsf.h5',input_shapes={'input_1':[1,320,240,3],})
# tflite_model = converter.convert()
# open("xx.tflite", "wb").write(tflite_model)
def iou(y_true,y_pred):
    y_true=(y_true>0.5)
    y_true=tf.keras.backend.cast(y_true, dtype='float32')
    return tf.keras.backend.sum(y_true*y_pred)/(tf.keras.backend.sum(y_pred),tf.keras.backend.sum(y_true)-tf.keras.backend.sum(y_true*y_pred)+0.0001)
def picture(pre_numpy,img_numpy,mask_numpy):
    #pre_numpy has shape num,height,width,channel
    voc_colormap=np.array([[0, 0, 0], [245,222,179]])
    num=len(img_numpy)
    target=(pre_numpy>0.5).squeeze().astype(int)
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    img=img_numpy*std+mean
    img=img.squeeze()
    mask=voc_colormap[mask_numpy.squeeze().astype(int)]
    tar=voc_colormap[target]/255.
    tmp=np.concatenate((img,tar,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    # plt.savefig('true_weight_alpha=05')
    plt.show()
def test(model):
    img=[]
    pred=[]
    mask=[]
    l5_list=[]
    for image,mask_img in val_data:
        l5=model.predict(image)
        label=(l5>0.5).squeeze().astype(int)
        l5_list.append(l5)
        pred.append(label)
        img.append(image)
        mask.append(mask_img)
    return np.concatenate(img,axis=0),np.concatenate(pred,axis=0),np.concatenate(mask,axis=0),l5_list
def thread(l5,mask):
    best_iou=[0,0]
    best_t=0
    best_label=0
    for i in np.arange(0.1,0.9,0.01):
        label=(l5>i).astype(int)
        ap,iou,hist,tmp=label_acc_score(mask,label,2)
        if iou[1]>best_iou[1]:
            best_iou=iou
            best_t=i
            best_label=label
    return  best_iou,best_t,best_label

model=keras.models.load_model('distill.h5')
tmp_model=Model(inputs=model.input,outputs=model.get_layer('stand').output)
img,pred,mask,l=test(tmp_model)
ap,iou,hist,tmp=label_acc_score(mask,pred,2)
# iou,thread,pred=thread(np.concatenate(l,axis=0),mask)
picture(pred[0:4],img[0:4],mask[:4])


# tmp_model=Model(inputs=model.input,outputs=model.get_layer('up_sampling2d_2').output)
# label=[]
# for i,j in train_data:
#     t=tmp_model.predict(i)
#     label.append(t)
# label=np.concatenate(label,axis=0)
# np.save('logits.npy',label)

# a,b=next(iter(val_data))
# label_a=tmp_model.predict(a)
# pred_a=(label_a>0.2).squeeze().astype(int)
# picture(pred_a[:4],img[:4],mask[:4])



# deeplab_model = keras.models.load_model('dddtrue_weight_dsf.hdf5',custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling })