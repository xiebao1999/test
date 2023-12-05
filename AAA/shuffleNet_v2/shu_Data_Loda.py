import glob
import tensorflow as tf
import numpy as np

'''图片预处理'''
def load_preprosess_image(img_path,labels):
    img_raw = tf.io.read_file(img_path)
    #img_tensor = tf.image.decode_png(img_raw,channels=1)  # img_tensor 变成一个像素三个通道为一组 数据
    img_tensor = tf.image.decode_jpeg(img_raw, channels=1)  # img_tensor 变成一个像素三个通道为一组 数据
    img_tensor = tf.image.resize(img_tensor,[224,224])  # 剪裁图像大小
    #img_tensor = tf.image.resize_with_crop_or_pad(img_tensor,224,224)  #oulu 剪裁
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255  # 标准化(归一化)
    labels = tf.reshape(labels,[1])
    return img,labels
def load_preprosess_test_image(img_path,labels):
    img_raw = tf.io.read_file(img_path)
    #img_tensor = tf.image.decode_png(img_raw,channels=1)  # img_tensor 变成一个像素三个通道为一组 数据
    img_tensor = tf.image.decode_jpeg(img_raw, channels=1)  # img_tensor 变成一个像素三个通道为一组 数据
    img_tensor = tf.image.resize(img_tensor,[224,224])  # 剪裁图像大小
    #img_tensor = tf.image.resize_with_crop_or_pad(img_tensor,224,224)   #oulu 剪裁
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255  # 标准化
    labels = tf.reshape(labels,[1])
    return img,labels

'获取图片路径和标签'
# imgs_path = glob.glob('C:/Users/Administrator/PycharmProjects/xiebaolai/AAA/data/JAFFE224/*/*.jpg')
imgs_path = glob.glob('C:/Users/Administrator/PycharmProjects/xiebaolai/AAA/data/CK+224/*/*.png')
# imgs_path = glob.glob('C:/Users/Administrator/PycharmProjects/xiebaolai/AAA/data/oulu224/*/*.jpg')


print(len(imgs_path))
img_p = imgs_path[700]
print(img_p.split('\\')[1])
print(imgs_path)
all_labels_name = [img_p.split('\\')[1] for img_p in imgs_path]
label_names = np.unique(all_labels_name)
print(label_names)
label_to_index = dict((name, i) for i, name in enumerate(label_names))
index_to_label = dict((v, k) for k, v in label_to_index.items())

'创建dataset和图片处理函数'
all_labels = [label_to_index.get(name) for name in all_labels_name]

# np.random.seed(2021)
random_index = np.random.permutation(len(imgs_path))
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]

BATCH_SIZE = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE

def loda_train_ds(train_path, train_labels):
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
    train_ds = train_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
    return train_ds

def loda_test_ds(test_path, test_labels):
    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))
    test_ds = test_ds.map(load_preprosess_test_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    return test_ds

