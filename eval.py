import os
import argparse
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from src.nets import nets
from src.utils import utils




def parse_args():
    parser = argparse.ArgumentParser('training the network')
    parser.add_argument('--eval_dir', type=str, default='/home/sun/my/data/fetal/norm/', help='where training data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--cv', type=int, default=1, help='cv')
    parser.add_argument('--model', type=str, default='MSMHA_CNN', help='select model')
    parser.add_argument('--ckpt', type=str, default='models/', help='ckpt saved path')


    args, _ = parser.parse_known_args()
    return args


def eval():
    args = parse_args()
    model = 'nets.M_' + args.model
    network = eval(model)

    tf.compat.v1.disable_eager_execution()
    eval_data, eval_label = utils.load_data_eval_3D(args.eval_dir, args.cv)

    w = int(eval_data.shape[1])
    h = int(eval_data.shape[2])

    xs = tf.compat.v1.placeholder(tf.float32, shape=[None, w, h, 20, 1])

    logits_ = network(xs)

    prob = tf.nn.sigmoid(logits_)
    pred = tf.cast(tf.greater_equal(prob, tf.ones_like(prob) * 0.5), dtype=tf.float32)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        ckpt = tf.compat.v1.train.get_checkpoint_state(args.ckpt + args.model + '_' + str(args.cv) + '/')
        print (args.ckpt + args.model + '_' + str(args.cv) + '/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint file found')
            return

        print("data size:", eval_data.shape)
        print("label_map size:", eval_label.shape)

        ite = int(eval_data.shape[0])//args.batch_size

        print('*** ' + args.model + ' evaling begin ***')

        for ss in range(ite):
            batch_image, batch_gt = utils.get_batch(ss, args.batch_size, eval_data, eval_label)
            prob_, pred_ = sess.run([prob, pred], feed_dict={xs: batch_image})

            flag = (args.cv - 1) * 24 + ss + 1
            sd = 'maps/' + args.model + '_' + str(flag) + '.nii.gz'

            if args.dims==3:
                img = np.reshape(pred_, [256, 256, 20])
                img = np.transpose(img, [2, 1, 0])
            if args.dims==2:
                img = np.reshape(pred_, [20, 256, 256])
            I = sitk.GetImageFromArray(img, isVector=False)
            I.SetSpacing([1.172, 1.172, 3.5])
            sitk.WriteImage(I, sd)

            print(sd + ' has been saved !!!')


if __name__ == '__main__':
    eval()
