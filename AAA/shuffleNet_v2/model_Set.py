
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential,Input

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
def CBAM_block(cbam_feature,ratio=4):
    cbam_feature = channel_attenstion(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attenstion(inputs, ratio=0.25):
    '''ratio代表第一个全连接层下降通道数的倍数'''

    channel = inputs.shape[-1]  # 获取输入特征图的通道数

    # 分别对输出特征图进行全局最大池化和全局平均池化
    # [h,w,c]==>[None,c]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x_max = layers.Reshape([1, 1, -1])(x_max)  # -1代表自动寻找通道维度的大小
    x_avg = layers.Reshape([1, 1, -1])(x_avg)  # 也可以用变量channel代替-1

    # 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
    x_max = layers.Dense(channel * ratio)(x_max)
    x_avg = layers.Dense(channel * ratio)(x_avg)

    # relu激活函数
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)

    # 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)

    # 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
    x = layers.Add()([x_max, x_avg])

    # 经过sigmoid归一化权重
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重向量相乘，给每个通道赋予权重
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]

    return x


def spatial_attention(inputs):
    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    # keepdims=Fale那么[b,h,w,c]==>[b,h,w]
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # 在通道维度求最大值
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # axis也可以为-1

    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_max, x_avg])

    # 1*1卷积调整通道[b,h,w,1]
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)

    # sigmoid函数权重归一化
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重相乘
    x = layers.Multiply()([inputs, x])

    return x



def SE_block(input_feature, r=16):
    in_channel = input_feature.shape[-1]
    x = layers.GlobalAveragePooling2D()(input_feature)
    # (?, ?) -> (?, 1, 1, ?)
    x = layers.Reshape(target_shape=(1,1,in_channel))(x)
    # 用2个1x1卷积代替全连接
    x = layers.Conv2D(filters=in_channel // r,
                      kernel_size=1,
                      strides=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=in_channel,
                      kernel_size=1,
                      strides=1)(x)
    x = layers.Activation('softmax')(x)
    x = layers.Multiply()([input_feature, x])
    return x

def SK_block(input_feature):
    in_channel = input_feature.shape[-1]

    U1 = layers.DepthwiseConv2D(
                               kernel_size=(3, 3),  # 卷积核个数默认3*3
                               strides=1,  # 步长
                               use_bias=False,  # 有BN层就不需要偏置
                               padding='same')(input_feature)  # 卷积过程中，步长=1，size不变；

    U2 = layers.Conv2D(in_channel,  # 卷积核个数（即通道数）下降到起始状态
                       kernel_size=(3, 3),  # 卷积核大小为1*1
                       strides=(1, 1),  # 步长为1*1
                       padding='same',
                       dilation_rate=(1,1),
                       use_bias=False)(input_feature)  # 有BN层就不需要偏置
    # Fuse
    U = U1+U2
    S = layers.GlobalAveragePooling2D()(U)
    print(S)
    S = tf.reshape(S,[-1,1,1,in_channel])
    print(S)
    Z =layers.Conv2D(32,1,strides=1, padding='same')(S)
    Z =layers.BatchNormalization()(Z)
    Z = layers.ReLU()(Z)
    print(Z)
    a = layers.Conv2D(in_channel,1,strides=1, padding='same')(Z)
    b = layers.Conv2D(in_channel,1,strides=1, padding='same')(Z)
    print(a,b)
    combine = tf.concat([a,b],1)
    print(combine)
    combine = layers.Activation('softmax')(combine)
    print(combine)
    a,b = tf.split(combine,num_or_size_splits=2, axis=1)
    print(a,b)
    V = a*U1+b*U2
    print(V)
    return V


