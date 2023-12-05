from tensorflow import keras
from sklearn.metrics import classification_report
from shu_Data_Loda import *
from shu_cm import *
from sklearn.model_selection import KFold

# 网络实现
import tensorflow as tf
from tensorflow.keras import layers, Model
from model_Set import SE_block

# 卷积模块
class ConvBNSiLU(layers.Layer):
    def __init__(self,
                 filters: int = 1,
                 kernel_size: int = 1,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super(ConvBNSiLU, self).__init__(**kwargs)

        # 卷积模块
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                  name="conv1")
        # 归一化
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")
        # Silu激活函数
        self.silu = tf.keras.activations.swish

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs) # 卷积
        x = self.bn(x, training=training) # 归一化
        x = self.silu(x) # relu激活
        return x

# 深度可分离卷积
class DWConvBN(layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super(DWConvBN, self).__init__(**kwargs)
        # 深度可分离卷积
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                              name="dw1")
        # 归一化
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")

    def call(self, inputs, training=None, **kwargs):
        x = self.dw_conv(inputs) #深度可分离卷积
        x = self.bn(x, training=training)
        return x

# shuffle操作
class ChannelShuffle(layers.Layer):
    def __init__(self, shape, groups: int = 2, **kwargs): # shape:传入的形状，批次高宽通道
        super(ChannelShuffle, self).__init__(**kwargs)
        batch_size, height, width, num_channels = shape # 获取输入信息
        assert num_channels % 2 == 0
        channel_per_group = num_channels // groups # 分组，每组通道数量

        # Tuple of integers, does not include the samples dimension (batch size).
        self.reshape1 = layers.Reshape((height, width, groups, channel_per_group)) # 根据分组后，每个组对应若干个通道
        self.reshape2 = layers.Reshape((height, width, num_channels)) # 将分组后的通道还原形状

    def call(self, inputs, **kwargs):
        x = self.reshape1(inputs) # 变形
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3]) # shuffle
        x = self.reshape2(x) # 变形
        return x

# 通道切分，默认分为两支
class ChannelSplit(layers.Layer):
    def __init__(self, num_splits: int = 2, **kwargs):
        super(ChannelSplit, self).__init__(**kwargs)
        self.num_splits = num_splits

    def call(self, inputs, **kwargs):
        splits = tf.split(inputs,
                          num_or_size_splits=self.num_splits,
                          axis=-1)
        return splits

# s1块
def shuffle_block_s1(inputs, output_c: int, stride: int, prefix: str):
    if stride != 1:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    branch_c = output_c // 2 # 输出多少通道，卷积核就是多少数量

    splits = ChannelSplit(num_splits=2,name=prefix + "/split")(inputs) # 通道切分处理，x1捷径分支，x2待处理分支
    x1, x2 = splits
    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)

    # main branch
    x2 = ConvBNSiLU(filters=branch_c, name=prefix + "/b2_conv1")(x2) #1*1卷积
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2) #dw卷积
    x2 = ConvBNSiLU(filters=branch_c, name=prefix + "/b2_conv2")(x2) #1*1卷积
    x2 = SE_block(x2) #se注意力机制

    x3 = ConvBNSiLU(filters=branch_c, name=prefix + "/b3_conv1")(x1) #1*1卷积

    x4 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b4_dw1")(x1)
    x4 = ConvBNSiLU(filters=branch_c, name=prefix + "/b4_conv1")(x1) #1*1卷积

    x5 = x1
    x = layers.Add()([x3,x4,x5])
    x = layers.Concatenate(name=prefix + "/concat")([x, x2]) #主、副分支拼接
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x) #channelShuffle处理

    return x

# s2块
def shuffle_block_s2(inputs, output_c: int, stride: int, prefix: str):
    if stride != 2:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    branch_c = output_c // 2

    # shortcut branch
    x1 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b1_dw1")(inputs)
    x1 = ConvBNSiLU(filters=branch_c, name=prefix + "/b1_conv1")(x1)
    x1 = SE_block(x1)

    # main branch
    x2 = ConvBNSiLU(filters=branch_c, name=prefix + "/b2_conv1")(inputs)
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
    x2 = ConvBNSiLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)
    x2 = SE_block(x2)

    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)

    return x

# 网络结构
def shufflenet_v2(num_classes: int, # 分类的类别个数
                  input_shape: tuple, # 输入形状
                  stages_repeats: list, # 每个阶段网络重复次数
                  stages_out_channels: list): # 每个阶段输出通道
    img_input = layers.Input(shape=input_shape)
    if len(stages_repeats) != 3: # 网络具有三个阶段，不是三个阶段,则报错体系
        raise ValueError("expected stages_repeats as list of 3 positive ints")
    if len(stages_out_channels) != 5: # 网络具有五个输出形状，不是五个则，报错体系
        raise ValueError("expected stages_out_channels as list of 5 positive ints")

    # 卷积
    x = ConvBNSiLU(filters=stages_out_channels[0],
                   kernel_size=3,
                   strides=2,
                   name="conv1")(img_input)
    # 最大池化
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2,
                            padding='same',
                            name="maxpool")(x)
    # 遍历阶段
    stage_name = ["stage{}".format(i) for i in [2, 3, 4]]
    for name, repeats, output_channels in zip(stage_name,
                                              stages_repeats,
                                              stages_out_channels[1:]):
        # 针对每一个阶段，先s2，后面都是s1
        for i in range(repeats):
            if i == 0:
                x = shuffle_block_s2(x, output_c=output_channels, stride=2, prefix=name + "_{}".format(i))
            else:
                x = shuffle_block_s1(x, output_c=output_channels, stride=1, prefix=name + "_{}".format(i))
        # 每一个阶段增加SE_block
        x = SE_block(x)
    # 卷积
    x = ConvBNSiLU(filters=stages_out_channels[-1], name="conv5")(x)
    # 全局池化
    x = layers.GlobalAveragePooling2D(name="globalpool")(x)
    # 全连接
    x = layers.Dense(units=num_classes, name="fc")(x)
    # 激活
    x = layers.Softmax()(x)

    model = Model(img_input, x, name="ShuffleNetV2_1.0")

    return model

# 搭建网络
def shufflenet_v2_x1_0(num_classes=7, input_shape=(224, 224, 1)):
    # 权重链接: https://pan.baidu.com/s/1M2mp98Si9eT9qT436DcdOw  密码: mhts
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[2, 3, 2],
                          stages_out_channels=[32, 224, 488, 976, 1024])
    return model

shufflenet_v2_x1_0().summary()

mean_acc = []
mean_loss = []


number = 1
kf = KFold(n_splits=5, random_state=7, shuffle=True)
for train_index, test_index in kf.split(imgs_path):
    tf.keras.backend.clear_session()
    m = shufflenet_v2_x1_0(input_shape=(224, 224, 1),
                               num_classes=7)
    test_labels = []
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = imgs_path[train_index], imgs_path[test_index]
    Y_train, Y_test = all_labels[train_index], all_labels[test_index]
    train_ds = loda_train_ds(X_train, Y_train)
    test_ds = loda_test_ds(X_test, Y_test)
    # print(train_index, test_index)
    epochs = 100

    train_count = len(X_train)
    test_count = len(X_test)
    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = test_count // BATCH_SIZE

    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc']
              )

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./GoogleNet{number}.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]
    history = m.fit(train_ds,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_ds,
                    validation_steps=validation_steps,
                    callbacks=callbacks
                    )
    evaluate_loss, evaluate_acc = m.evaluate(test_ds)
    print('Model_loss: ', evaluate_loss, 'Model_acc', evaluate_acc)
    mean_acc.append(evaluate_acc)
    mean_loss.append(evaluate_loss)

    model = GoogleNet(class_num=7)
    model.build((None,224,224,3))
    weights_path = "./GoogleNet10.h5"
    m.load_weights(weights_path)

    predict_classes = m.predict(test_ds)  # 对测试数据集进行预测
    true_classes = np.argmax(predict_classes, 1)  # 汲取预测结果

    cmpic(true_classes, Y_test, name=number)

    # 绘图
    out_dir = r"shu_image_cm/tempt"

    # out_dir = r"image/ck+"
    plt.plot(history.epoch, history.history.get("acc"), label='acc')
    plt.plot(history.epoch, history.history.get("val_acc"), label='val_acc')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'Accuracy_' + str(number) + '.png'))
    plt.show()

    plt.plot(history.epoch, history.history.get("loss"), label='loss')
    plt.plot(history.epoch, history.history.get("val_loss"), label='val_loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'Loss_' + str(number) + '.png'))
    plt.show()

    number = number + 1

print('mean_acc:', sum(mean_acc) / len(mean_acc), 'mean_loss:', sum(mean_loss) / len(mean_loss))

predict_classes = m.predict(test_ds)  # 对测试数据集进行预测
true_classes = np.argmax(predict_classes, 1)  # 汲取预测结果
labels_name = ('anger','contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise') # ck+
# labels_name = ('anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise')  # jaffe
# labels_name = ('anger',  'disgust', 'fear', 'happy', 'sadness', 'surprise') # oulu
print(classification_report(true_classes, Y_test, target_names=labels_name))
cmpic(true_classes, Y_test, name='final')
print('model_evaluate:', m.evaluate(test_ds))