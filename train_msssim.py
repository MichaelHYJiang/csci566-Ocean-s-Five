#!/usr/bin/env python

from __future__ import division
import os, time, glob

import numpy as np
import tensorflow as tf
from network import network
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

def crop(raw, gt_raw, start_frame=0):
    # inputs must be in a form of [batch_num, frame_num, height, width, channel_num]
    tt = start_frame
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
    raw = np.expand_dims(np.float32(np.load(in_path) / 65535.0), axis=0)
    gt_raw = np.expand_dims(np.float32(np.load(gt_path) / 255.0), axis=0)

    input_patch, gt_patch = crop(raw, gt_raw, np.random.randint(ALL_FRAME - CROP_FRAME))
    input_patch = np.minimum(input_patch, 1.0)

    loss, output = sess.run([G_loss, out_image], feed_dict={in_image: input_patch, gt_image: gt_patch})
    return loss


def main():
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, CROP_FRAME, None, None, 4])
    gt_image = tf.placeholder(tf.float32, [None, CROP_FRAME, None, None, 3])
    out_image = network(in_image)
    if DEBUG:
        print '[DEBUG] out_image shape:', out_image.shape

    v_loss = tf.Variable(0.0)

    ## DEFINE LOSS HERE
    # L1-Loss
    # G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

    # SSIM Loss
    im_out = tf.image.convert_image_dtype(out_image, tf.float32)
    im_gt = tf.image.convert_image_dtype(gt_image, tf.float32)
    SSIM_Loss = 1 - tf.image.ssim(im_out, im_gt, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    SSIM_Loss = tf.reduce_mean(SSIM_Loss)

    # Multi-Scale-SSIM Loss
    SSIM_Multi_Loss = 1 - tf.image.ssim_multiscale(im_out, im_gt, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    SSIM_Multi_Loss = tf.reduce_mean(SSIM_Multi_Loss)


    ## CHOOSE THE TYPE OR COMBINATION OF LOSS(ES)
    # G_loss = SSIM_Loss
    G_loss = SSIM_Loss + SSIM_Multi_Loss

    # tensorboard summary
    tf.summary.scalar('loss', v_loss)
    # tf.summary.scalar('validation loss', v_loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, TRAIN_LOG_DIR), graph=tf.get_default_graph())
    writer_val = tf.summary.FileWriter(os.path.join(LOGS_DIR, VAL_LOG_DIR), graph=tf.get_default_graph())

    lr = tf.placeholder(tf.float32)
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

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

    np.random.seed(NUMPY_RANDOM_SEED)
    count = 0
    for epoch in range(lastepoch, MAX_EPOCH + 1):
        save_results = False
        if epoch % SAVE_FREQ == 0:
            save_results = True

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

                # read data

                input_image_read_in = np.load(in_path)
                input_images[ind] = np.expand_dims(np.float32(input_image_read_in / 65535.0), axis=0)
                raw = input_images[ind]

                gt_image_read_in = np.load(gt_path)
                gt_images[ind] = np.expand_dims(np.float32(gt_image_read_in / 255.0), axis=0)
                gt_raw = gt_images[ind]

                input_patch, gt_patch = crop(raw, gt_raw, start_frame)

                input_patch, gt_patch = flip(input_patch, gt_patch)

                input_patch = np.minimum(input_patch, 1.0)

                # calculate and optimize loss

                _, G_current, output = sess.run([G_opt, G_loss, out_image], feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
                output = np.minimum(np.maximum(output, 0), 1)
                g_loss[ind] = G_current

                # save loss
                summary = sess.run(summary_op, feed_dict={v_loss:G_current})
                writer.add_summary(summary, count)
                count += 1

                # std output
                print("%d %d Loss=%.8f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st)), train_id
            
        # validation after each epoch
        v_start = time.time()
        losses = []
        for i in range(len(valid_in_files)):
            in_path = valid_in_files[i]
            gt_path = valid_gt_files[i]
            loss = validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image)
            losses.append(loss)
        losses = np.array(losses)
        summary = sess.run(summary_op, feed_dict={v_loss:np.mean(losses)})
        writer_val.add_summary(summary, count)
        print 'validation: Loss={:.8f} Time={:.3f}s'.format(np.mean(losses), time.time() - v_start)

        saver.save(sess, CHECKPOINT_DIR + 'model.ckpt')
        if save_results:
            saver.save(sess, RESULT_DIR + '%04d/' % epoch + 'model.ckpt')


if __name__ == '__main__':
    main()
