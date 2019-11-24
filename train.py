#!/usr/bin/env python

# ----------------------------------------------------------------
# 3D-Conv-2D-Pool-UNet Training
# Written by Haiyang Jiang
# Mar 1st 2019
# ----------------------------------------------------------------

from __future__ import division
import os, time, glob

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skvideo.io import vwrite

from network import network, loss_network
from config import *

# get train IDs
with open(FILE_LIST) as f:
    text = f.readlines()
train_files = text

train_ids = [line.strip().split(' ')[0] for line in train_files]
gt_files = [line.strip().split(' ')[1] for line in train_files]
in_files = [line.strip().split(' ')[2] for line in train_files]

# get validation set IDs
with open(VALID_LIST) as f:
    text = f.readlines()
validate_files = text

valid_ids = [line.strip().split(' ')[0] for line in validate_files]
valid_gt_files = [line.strip().split(' ')[1] for line in validate_files]
valid_in_files = [line.strip().split(' ')[2] for line in validate_files]


raw = np.load(in_files[0])
F = raw.shape[0]
H = raw.shape[1]
W = raw.shape[2]

if DEBUG:
    print '[DEBUG] input shape:', F, H, W
    SAVE_FREQ = 2
    train_ids = train_ids[0:250]
    print len(train_ids)
    MAX_EPOCH = 50


def demosaic(in_vid, converter=cv2.COLOR_BayerGB2BGR):
    bayer_input = np.zeros([in_vid.shape[0], in_vid.shape[1] * 2, in_vid.shape[2] * 2])
    bayer_input[:, ::2, ::2] = in_vid[:, :, :, 0]
    bayer_input[:, ::2, 1::2] = in_vid[:, :, :, 1]
    bayer_input[:, 1::2, ::2] = in_vid[:, :, :, 2]
    bayer_input[:, 1::2, 1::2] = in_vid[:, :, :, 3]
    bayer_input = (bayer_input * 65535).astype('uint16')
    rgb_input = np.zeros([bayer_input.shape[0], bayer_input.shape[1], bayer_input.shape[2], 3])
    for j in range(bayer_input.shape[0]):
        rgb_input[j] = cv2.cvtColor(bayer_input[j], converter)
    return rgb_input / 65535.0


def crop(raw, gt_raw, start_frame=0):
    # inputs must be in a form of [batch_num, frame_num, height, width, channel_num]
    tt = start_frame
    W = raw.shape[3]
    H = raw.shape[2]
    xx = np.random.randint(0, W - CROP_WIDTH)
    yy = np.random.randint(0, H - CROP_HEIGHT)

    input_patch = raw[:, tt:tt + CROP_FRAME, yy:yy + CROP_HEIGHT, xx:xx + CROP_WIDTH, :]
    gt_patch = gt_raw[:, tt:tt + CROP_FRAME, yy * 2:(yy + CROP_HEIGHT) * 2, xx * 2:(xx + CROP_WIDTH) * 2, :]
    return input_patch, gt_patch


def flip(input_patch, gt_patch):
    # inputs must be in a form of [batch_num, frame_num, height, width, channel_num]
    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=3)
        gt_patch = np.flip(gt_patch, axis=3)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 1, 3, 2, 4))
        gt_patch = np.transpose(gt_patch, (0, 1, 3, 2, 4))
    return input_patch, gt_patch


def validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image):
    read_in = np.load(in_path)
    
    # 16 bit
    raw = np.expand_dims(read_in / 65535.0, axis=0)

    gt_raw = np.expand_dims(np.float32(np.load(gt_path) / 255.0), axis=0)

    input_patch, gt_patch = crop(raw, gt_raw, np.random.randint(ALL_FRAME - CROP_FRAME))

    input_patch, gt_patch = flip(input_patch, gt_patch)
    input_patch = np.minimum(input_patch, 1.0)
    loss = sess.run(G_loss, feed_dict={in_image: input_patch, gt_image: gt_patch})
    return loss


def sigmoid_cross_entropy_loss(labels, logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
    return tf.reduce_mean(loss)


def main():
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [1, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 4])
    gt_image = tf.placeholder(tf.float32, [1, CROP_FRAME, CROP_HEIGHT * 2, CROP_WIDTH * 2, 3])
    out_image = network(in_image)
    if DEBUG:
        print '[DEBUG] out_image shape:', out_image.shape

    logits_out_images = loss_network(out_image[0])[:, 0]
    logits_gt_images = loss_network(gt_image[0])[:, 0]
    # labels = [0] * CROP_FRAME + [1] * CROP_FRAME
    label_ones = tf.constant([0.0] * CROP_FRAME)
    label_zeros = tf.constant([1.0] * CROP_FRAME)
    dis_loss_op = sigmoid_cross_entropy_loss(label_ones, logits_gt_images) + sigmoid_cross_entropy_loss(label_zeros, logits_out_images)
    gen_loss_op = sigmoid_cross_entropy_loss(label_ones, logits_out_images)
    # dis_loss_op = -tf.reduce_mean(tf.abs(logits_out_images - logits_gt_images))
    # gen_loss_op = -dis_loss_op
    
    
    D_loss =  dis_loss_op #1e-7 * tf.reduce_mean(tf.abs(loss_out_image - loss_gt_image))
    L1_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
    G_loss = gen_loss_op + L1_loss
    v_loss = tf.Variable(0.0)

    # tensorboard summary
    tf.summary.scalar('loss', v_loss)
    # tf.summary.scalar('validation loss', v_loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, TRAIN_LOG_DIR, 'l1_loss'), graph=tf.get_default_graph())
    writer_g = tf.summary.FileWriter(os.path.join(LOGS_DIR, TRAIN_LOG_DIR, 'g_loss'), graph=tf.get_default_graph())
    writer_d = tf.summary.FileWriter(os.path.join(LOGS_DIR, TRAIN_LOG_DIR, 'd_loss'), graph=tf.get_default_graph())
    writer_val = tf.summary.FileWriter(os.path.join(LOGS_DIR, VAL_LOG_DIR), graph=tf.get_default_graph())

    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32)
    dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dis')
    gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gen')
    
    D_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=dis_var_list)
    
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=gen_var_list)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images = [None] * len(train_ids)
    input_images = [None] * len(train_ids)

    g_loss = np.zeros((len(train_ids), 1))

    lastepoch = 0
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    else:
        all_items = glob.glob(os.path.join(RESULT_DIR, '*'))
        all_folders = [os.path.basename(d) for d in all_items if os.path.isdir(d) and os.path.basename(d).isdigit()]
        for folder in all_folders:
            lastepoch = np.maximum(lastepoch, int(folder))

    learning_rate = INIT_LR

    np.random.seed(ord('c') + 137)
    count = 0
    for epoch in range(lastepoch, MAX_EPOCH + 1):
        if epoch % SAVE_FREQ == 0:
            save_results = True
            if not os.path.isdir(RESULT_DIR + '%04d' % epoch):
                os.makedirs(RESULT_DIR + '%04d' % epoch)
        else:
            save_results = False
        cnt = 0
        if epoch > DECAY_EPOCH:
            learning_rate = DECAY_LR

        N = len(train_ids)
        all_order = np.random.permutation(N)
        last_group = (N // GROUP_NUM) * GROUP_NUM
        split_order = np.split(all_order[:last_group], (N // GROUP_NUM))
        split_order.append(all_order[last_group:])
        for order in split_order:
            gt_images = [None] * len(train_ids)
            input_images = [None] * len(train_ids)
            order_frame = [(one, y) for y in [t for t in np.random.permutation(ALL_FRAME - CROP_FRAME) if t % FRAME_FREQ == 0] for one in order]

            index = np.random.permutation(len(order_frame))
            for idx in index:
                ind, start_frame = order_frame[idx]
                start_frame += np.random.randint(FRAME_FREQ)
                # get the path from image id
                train_id = train_ids[ind] + '_start_frame_' + str(start_frame)
                in_path = in_files[ind]

                gt_path = gt_files[ind]

                st = time.time()
                cnt += 1

                if input_images[ind] is None:
                    read_in = np.load(in_path)
                    # 16 bit
                    input_images[ind] = np.expand_dims(read_in / 65535.0, axis=0)
                raw = input_images[ind]
                # raw = np.expand_dims(raw / 65535.0, axis=0)

                if gt_images[ind] is None:
                    gt_images[ind] = np.expand_dims(np.float32(np.load(gt_path) / 255.0), axis=0)
                gt_raw = gt_images[ind]
                # gt_raw = np.expand_dims(np.float32(gt_raw / 255.0), axis=0)

                input_patch, gt_patch = crop(raw, gt_raw, start_frame)

                input_patch, gt_patch = flip(input_patch, gt_patch)


                input_patch = np.minimum(input_patch, 1.0)

                # output0 = sess.run(out_image, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                _, G_current_d = sess.run([D_opt, D_loss], feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                # output1 = sess.run(out_image, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                # assert 0.0 == abs(output0 - output1).sum()
                # lg0 = sess.run(logits_gt_images, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                _, G_current_g, output = sess.run([G_opt, G_loss, out_image], feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                G_current = sess.run(L1_loss, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                # lo = sess.run(logits_out_images, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                # lg = sess.run(logits_gt_images, feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                # assert 0.0 == abs(lg - lg0).sum()
                output = np.minimum(np.maximum(output, 0), 1)
                g_loss[ind] = G_current




                # save loss
                summary = sess.run(summary_op, feed_dict={v_loss:G_current})
                writer.add_summary(summary, count)
                summary = sess.run(summary_op, feed_dict={v_loss:G_current_g})
                writer_g.add_summary(summary, count)
                summary = sess.run(summary_op, feed_dict={v_loss:G_current_d})
                writer_d.add_summary(summary, count)
                count += 1

                if save_results and start_frame in SAVE_FRAMES:
                    temp = np.concatenate((gt_patch[0, :, ::-1, :, :], output[0, :, ::-1, :, :]), axis=2)
                    try:
                        vwrite((RESULT_DIR + '%04d/%s_train.avi' % (epoch, train_id)), (temp * 255).astype('uint8'))
                    except OSError as e:
                        print('\t', e, 'Skip saving.')



                print("%d %d Loss=%.8f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st)), train_id, '\n', G_current_d, G_current_g - G_current, G_current, G_current_g#, abs(lg0 - lg).mean()#, '\n', lo, '\n', lg, '\n', '=' * 40
            
        # validation after each epoch
        v_start = time.time()
        losses = []
        for i in range(len(valid_in_files)):
            in_path = valid_in_files[i]
            gt_path = valid_gt_files[i]
            loss = validate(in_path, gt_path, sess, L1_loss, out_image, in_image, gt_image)
            losses += loss,
        summary = sess.run(summary_op, feed_dict={v_loss:np.mean(losses)})
        writer_val.add_summary(summary, count)
        print 'validation: Loss={:.8f} Time={:.3f}s'.format(np.mean(losses), time.time() - v_start)

        saver.save(sess, CHECKPOINT_DIR + 'model.ckpt')
        if save_results:
            saver.save(sess, RESULT_DIR + '%04d/' % epoch + 'model.ckpt')


if __name__ == '__main__':
    main()
