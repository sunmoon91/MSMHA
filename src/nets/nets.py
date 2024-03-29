import tensorflow as tf
from tensorflow.keras.layers import *

def weight_variable(shape, stddev=0.1):
    initial = tf.random.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def conv3d(x, W):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
    return conv_3d



def deconv3d(x, W, output_shape, stride, padding='SAME'):
    return tf.nn.conv3d_transpose(x, W, output_shape, [1, stride, stride, stride, 1], padding=padding)



def max_pool3d(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')



def crop_and_concat3d(x1, x2):
    return tf.concat([x1, x2], 4)


def crop_and_concat3(x1, x2, x3,):
    return tf.concat([x1, x2, x3], 4)


def crop_and_concat4(x1, x2, x3, x4):
    return tf.concat([x1, x2, x3, x4], 4)


def batchnorm3d(x, is_training, name=None):
    bn = BatchNormalization()(x, training=is_training)
    return bn




# ###############################
# ######### MSMHA_CNN ############
# ###############################

def M_MSMHA_CNN(xs, is_training=True):
    w_i_1_1 = weight_variable([1, 1, 1, 1, 32])
    b_i_1_1 = bias_variable([32])
    bn_i_1_1 = tf.nn.relu(
        batchnorm3d(conv3d(xs, w_i_1_1) + b_i_1_1, is_training=is_training, name='layer_i_1_1'))

    w_i_2_1 = weight_variable([3, 3, 1, 32, 32])
    b_i_2_1 = bias_variable([32])
    bn_i_2_1 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_1_1, w_i_2_1) + b_i_2_1, is_training=is_training, name='layer_i_2_1'))

    w_i_1_2 = weight_variable([1, 1, 1, 1, 32])
    b_i_1_2 = bias_variable([32])
    bn_i_1_2 = tf.nn.relu(
        batchnorm3d(conv3d(xs, w_i_1_2) + b_i_1_2, is_training=is_training, name='layer_i_1_2'))

    w_i_2_2 = weight_variable([3, 3, 3, 32, 32])
    b_i_2_2 = bias_variable([32])
    bn_i_2_2 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_1_2, w_i_2_2) + b_i_2_2, is_training=is_training, name='layer_i_2_2'))

    pool_i_1_3 = tf.nn.avg_pool3d(xs, ksize=[1, 3, 3, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_2_3 = weight_variable([1, 1, 1, 1, 32])
    b_i_2_3 = bias_variable([32])
    bn_i_2_3 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_1_3, w_i_2_3) + b_i_2_3, is_training=is_training, name='layer_i_2_3'))

    pool_i_1_4 = tf.nn.avg_pool3d(xs, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_2_4 = weight_variable([1, 1, 1, 1, 32])
    b_i_2_4 = bias_variable([32])
    bn_i_2_4 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_1_4, w_i_2_4) + b_i_2_4, is_training=is_training, name='layer_i_2_4'))

    o_2 = crop_and_concat4(bn_i_2_1, bn_i_2_2, bn_i_2_3, bn_i_2_4)

    w_o_2 = weight_variable([1, 1, 1, 128, 32])
    b_o_2 = bias_variable([32])
    c_o_2 = tf.nn.relu(batchnorm3d(conv3d(o_2, w_o_2) + b_o_2, is_training=is_training, name='layer_c_2'))
    w_o_2_ = weight_variable([1, 1, 1, 32, 128])
    b_o_2_ = bias_variable([128])
    scale_2 = tf.nn.sigmoid(conv3d(c_o_2, w_o_2_) + b_o_2_)
    se_2 = o_2 * scale_2

    pool_i_2 = max_pool3d(se_2, 2)

    # block2

    w_i_3_1 = weight_variable([1, 1, 1, 128, 64])
    b_i_3_1 = bias_variable([64])
    bn_i_3_1 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_2, w_i_3_1) + b_i_3_1, is_training=is_training, name='layer_i_3_1'))

    w_i_4_1 = weight_variable([3, 3, 1, 64, 64])
    b_i_4_1 = bias_variable([64])
    bn_i_4_1 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_3_1, w_i_4_1) + b_i_4_1, is_training=is_training, name='layer_i_4_1'))

    w_i_3_2 = weight_variable([1, 1, 1, 128, 64])
    b_i_3_2 = bias_variable([64])
    bn_i_3_2 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_2, w_i_3_2) + b_i_3_2, is_training=is_training, name='layer_i_3_2'))

    w_i_4_2 = weight_variable([3, 3, 3, 64, 64])
    b_i_4_2 = bias_variable([64])
    bn_i_4_2 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_3_2, w_i_4_2) + b_i_4_2, is_training=is_training, name='layer_i_4_2'))

    pool_i_3_3 = tf.nn.avg_pool3d(pool_i_2, ksize=[1, 3, 3, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_4_3 = weight_variable([1, 1, 1, 128, 64])
    b_i_4_3 = bias_variable([64])
    bn_i_4_3 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_3_3, w_i_4_3) + b_i_4_3, is_training=is_training, name='layer_i_4_3'))

    pool_i_3_4 = tf.nn.avg_pool3d(pool_i_2, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_4_4 = weight_variable([1, 1, 1, 128, 64])
    b_i_4_4 = bias_variable([64])
    bn_i_4_4 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_3_4, w_i_4_4) + b_i_4_4, is_training=is_training, name='layer_i_4_4'))

    o_4 = crop_and_concat4(bn_i_4_1, bn_i_4_2, bn_i_4_3, bn_i_4_4)

    w_o_4 = weight_variable([1, 1, 1, 256, 64])
    b_o_4 = bias_variable([64])
    c_o_4 = tf.nn.relu(batchnorm3d(conv3d(o_4, w_o_4) + b_o_4, is_training=is_training, name='layer_c_4'))
    w_o_4_ = weight_variable([1, 1, 1, 64, 256])
    b_o_4_ = bias_variable([256])
    scale_4 = tf.nn.sigmoid(conv3d(c_o_4, w_o_4_) + b_o_4_)

    se_4 = o_4 * scale_4

    pool_i_4 = max_pool3d(se_4, 2)

    # block3

    w_i_5_1 = weight_variable([1, 1, 1, 256, 128])
    b_i_5_1 = bias_variable([128])
    bn_i_5_1 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_4, w_i_5_1) + b_i_5_1, is_training=is_training, name='layer_i_5_1'))

    w_i_6_1 = weight_variable([3, 3, 1, 128, 128])
    b_i_6_1 = bias_variable([128])
    bn_i_6_1 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_5_1, w_i_6_1) + b_i_6_1, is_training=is_training, name='layer_i_6_1'))

    w_i_5_2 = weight_variable([1, 1, 1, 256, 128])
    b_i_5_2 = bias_variable([128])
    bn_i_5_2 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_4, w_i_5_2) + b_i_5_2, is_training=is_training, name='layer_i_5_2'))

    w_i_6_2 = weight_variable([3, 3, 3, 128, 128])
    b_i_6_2 = bias_variable([128])
    bn_i_6_2 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_5_2, w_i_6_2) + b_i_6_2, is_training=is_training, name='layer_i_6_2'))

    pool_i_5_3 = tf.nn.avg_pool3d(pool_i_4, ksize=[1, 3, 3, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_6_3 = weight_variable([1, 1, 1, 256, 128])
    b_i_6_3 = bias_variable([128])
    bn_i_6_3 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_5_3, w_i_6_3) + b_i_6_3, is_training=is_training, name='layer_i_6_3'))

    pool_i_5_4 = tf.nn.avg_pool3d(pool_i_4, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    w_i_6_4 = weight_variable([1, 1, 1, 256, 128])
    b_i_6_4 = bias_variable([128])
    bn_i_6_4 = tf.nn.relu(
        batchnorm3d(conv3d(pool_i_5_4, w_i_6_4) + b_i_6_4, is_training=is_training, name='layer_i_6_4'))

    o_6 = crop_and_concat4(bn_i_6_1, bn_i_6_2, bn_i_6_3, bn_i_6_4)

    w_o_6 = weight_variable([1, 1, 1, 512, 128])
    b_o_6 = bias_variable([128])
    c_o_6 = tf.nn.relu(batchnorm3d(conv3d(o_6, w_o_6) + b_o_6, is_training=is_training, name='layer_c_6'))
    w_o_6_ = weight_variable([1, 1, 1, 128, 512])
    b_o_6_ = bias_variable([512])
    scale_6 = tf.nn.sigmoid(conv3d(c_o_6, w_o_6_) + b_o_6_)

    se_6 = o_6 * scale_6

    # decoding
    w_d_i_7 = weight_variable([3, 3, 3, 256, 512])
    b_d_i_7 = bias_variable([256])
    deconv_i_7 = tf.nn.relu(deconv3d(se_6, w_d_i_7, tf.shape(se_4), 2) + b_d_i_7)

    # block

    crop8 = crop_and_concat3d(deconv_i_7, se_4)

    w_i_8 = weight_variable([3, 3, 3, 512, 256])
    b_i_8 = bias_variable([256])
    bn_i_8 = tf.nn.relu(batchnorm3d(conv3d(crop8, w_i_8) + b_i_8, is_training=is_training, name='layer_i_8'))

    w_i_9 = weight_variable([3, 3, 3, 256, 256])
    b_i_9 = bias_variable([256])
    bn_i_9 = tf.nn.relu(batchnorm3d(conv3d(bn_i_8, w_i_9) + b_i_9, is_training=is_training, name='layer_i_9'))

    # decoding
    w_d_i_10 = weight_variable([3, 3, 3, 128, 256])
    b_d_i_10 = bias_variable([128])
    deconv_i_10 = tf.nn.relu(deconv3d(bn_i_9, w_d_i_10, tf.shape(se_2), 2) + b_d_i_10)

    crop_11 = crop_and_concat3d(deconv_i_10, se_2)

    w_i_11 = weight_variable([3, 3, 3, 256, 128])
    b_i_11 = bias_variable([128])
    bn_i_11 = tf.nn.relu(
        batchnorm3d(conv3d(crop_11, w_i_11) + b_i_11, is_training=is_training, name='layer_i_11'))

    w_i_12 = weight_variable([3, 3, 3, 128, 64])
    b_i_12 = bias_variable([64])
    bn_i_12 = tf.nn.relu(
        batchnorm3d(conv3d(bn_i_11, w_i_12) + b_i_12, is_training=is_training, name='layer_i_12'))

    # FPN

    w_d_f_7 = weight_variable([3, 3, 3, 256, 512])
    b_d_f_7 = bias_variable([256])
    deconv_f_7 = tf.nn.relu(deconv3d(se_6, w_d_f_7, tf.shape(se_4), 2) + b_d_f_7)

    w_f_8 = weight_variable([1, 1, 1, 256, 256])
    b_f_8 = bias_variable([256])
    bn_f_8 = tf.nn.relu(
        batchnorm3d(conv3d(deconv_f_7, w_f_8) + b_f_8, is_training=is_training, name='layer_f_8'))

    w_d_f_9 = weight_variable([3, 3, 3, 128, 256])
    b_d_f_9 = bias_variable([128])
    deconv_f_9 = tf.nn.relu(deconv3d(bn_f_8, w_d_f_9, tf.shape(se_2), 2) + b_d_f_9)

    w_f_10 = weight_variable([1, 1, 1, 128, 64])
    b_f_10 = bias_variable([64])
    bn_f_10 = tf.nn.relu(
        batchnorm3d(conv3d(deconv_f_9, w_f_10) + b_f_10, is_training=is_training, name='layer_f_10'))

    # s2
    w_d_f_11 = weight_variable([3, 3, 3, 128, 256])
    b_d_f_11 = bias_variable([128])
    deconv_f_11 = tf.nn.relu(deconv3d(bn_i_9, w_d_f_11, tf.shape(se_2), 2) + b_d_f_11)

    w_f_12 = weight_variable([1, 1, 1, 128, 64])
    b_f_12 = bias_variable([64])
    bn_f_12 = tf.nn.relu(
        batchnorm3d(conv3d(deconv_f_11, w_f_12) + b_f_12, is_training=is_training, name='layer_f_12'))


    o_13 = crop_and_concat3(bn_f_10, bn_f_12, bn_i_12)


    w_o_13 = weight_variable([1, 1, 1, 192, 48])
    b_o_13 = bias_variable([48])
    c_o_13 = tf.nn.relu(batchnorm3d(conv3d(o_13, w_o_13) + b_o_13, is_training=is_training, name='layer_c_13'))
    w_o_13_ = weight_variable([1, 1, 1, 48, 192])
    b_o_13_ = bias_variable([192])
    scale_13 = tf.nn.sigmoid(conv3d(c_o_13, w_o_13_) + b_o_13_)
    se_13 = o_13 * scale_13

    w_i_15 = weight_variable([1, 1, 1, 192, 1])
    b_i_15 = bias_variable([1])
    c_i_15 = conv3d(se_13, w_i_15) + b_i_15

    return c_i_15
