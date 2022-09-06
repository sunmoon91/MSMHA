import os
import argparse
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from src.nets import nets
from src.utils import utils


def parse_args():
    parser = argparse.ArgumentParser('training the network')
    parser.add_argument('--data_dir', type=str, default='/home/sun/my/data/fetal/norm/', help='where data')
    parser.add_argument('--dataset', type=str, default='validation', help='testing data index')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--cv', type=int, default=1, help='cv')
    parser.add_argument('--model', type=str, default='MSMHV_CNN', help='select model')
    parser.add_argument('--ckpt', type=str, default='models/', help='ckpt saved path')

    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_args()
    model = 'nets.M_' + args.model
    network = eval(model)

    tf.compat.v1.disable_eager_execution()
    train_data, train_label = utils.load_data_train_3D(args.data_dir, args.cv)

    w = int(train_data.shape[1])
    h = int(train_data.shape[2])

    xs = tf.compat.v1.placeholder(tf.float32, shape=[None, w, h, 20, 1])
    ys = tf.compat.v1.placeholder(tf.float32, shape=[None, w, h, 20, 1])


    logits_ = network(xs)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=logits_))
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        ckpt = tf.compat.v1.train.get_checkpoint_state(args.ckpt + args.model + '_' + str(args.cv) + '/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint file found')

        print("data size:", train_data.shape)
        print("label_map size:", train_label.shape)

        ite = int(train_data.shape[0]) // args.batch_size

        print('*** ' + args.model + ' testing begin ***')

        for epoch in range(args.epoch):
            for ss in range(ite):
                batch_image, batch_gt = utils.get_batch(ss, args.batch_size, train_data, train_label)
                _, loss_ = sess.run([train_step, loss], feed_dict={xs: batch_image, ys: batch_gt})
                print('Train Epoch: ', epoch, 'Subject: ', ss, 'loss_: ', loss_)
        saver.save(sess, args.ckpt + args.model + '_' + str(args.cv) + '/model.ckpt')

if __name__ == '__main__':
    train()
