# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob

input_dir = './down_sampled_frame_pairs/input/'
gt_dir = './down_sampled_frame_pairs/gt/'
valid_input_dir = './down_sampled_frame_pairs/valid/input/'
valid_gt_dir = './down_sampled_frame_pairs/valid/gt/'
checkpoint_dir = './result_downsampled_he2he/'
result_dir = './result_downsampled_he2he/'
LOGS_DIR = result_dir

# get train IDs
train_fns = glob.glob(gt_dir + '*')
train_ids = [os.path.basename(train_fn) for train_fn in train_fns]

# get validation IDs
valid_fns = glob.glob(valid_gt_dir + '*')
valid_ids = [os.path.basename(valid_fn) for valid_fn in valid_fns]



DECAY_EPOCH = 30
MAX_EPOCH = 60

ps = 128  # patch size for training
save_freq = 5

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def pack_raw(raw):# not used
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image):
    input_images= np.load(in_path).astype('uint16')

    gt_images = np.load(gt_path).astype('uint8')

    # crop
    H = input_images.shape[0]
    W = input_images.shape[1]

    #print 'H, W:', H, W
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    
    raw = np.expand_dims(np.float32(input_images / 65535.0), axis=0)
    input_patch = raw[:, yy:yy + ps, xx:xx + ps, :]
    gt_raw = np.expand_dims(np.float32(gt_images / 255.0), axis=0)
    gt_patch = gt_raw[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=0)
        gt_patch = np.flip(gt_patch, axis=0)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

    input_patch = np.minimum(input_patch, 1.0)
    loss = sess.run(G_loss, feed_dict={in_image: input_patch, gt_image: gt_patch})
    return loss


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
v_loss = tf.Variable(0.0)

# tensorboard summary
tf.summary.scalar('loss', v_loss)
# tf.summary.scalar('validation loss', v_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, 'train'), graph=tf.get_default_graph())
writer_val = tf.summary.FileWriter(os.path.join(LOGS_DIR, 'val'), graph=tf.get_default_graph())

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
#gt_images = [None] * len(train_ids)
#input_images = [None] * len(train_ids)

g_loss = np.zeros((len(train_ids), 1))

allfolders = glob.glob('./result/*')
lastepoch = 0

learning_rate = 1e-4
lr_decayed = False
print '[INFO] initial learning rate:', learning_rate
count = 0
for epoch in range(lastepoch + 1, MAX_EPOCH + 1):
    cnt = 0
    e_st = time.time()
    if not lr_decayed and epoch > DECAY_EPOCH:
        learning_rate = 1e-5
        print '[INFO] decayed learning rate:', learning_rate
        lr_decayed = True

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_path = input_dir + train_id
        in_fn = os.path.basename(in_path)

        gt_path = gt_dir + train_id
        gt_fn = os.path.basename(gt_path)

        st = time.time()
        cnt += 1

        #if input_images[ind] is None:
        input_images= np.load(in_path).astype('uint16')

        gt_images = np.load(gt_path).astype('uint8')

        # crop
        H = input_images.shape[0]
        W = input_images.shape[1]

        #print 'H, W:', H, W
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        
        raw = np.expand_dims(np.float32(input_images / 65535.0), axis=0)
        input_patch = raw[:, yy:yy + ps, xx:xx + ps, :]
        gt_raw = np.expand_dims(np.float32(gt_images / 255.0), axis=0)
        gt_patch = gt_raw[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current

        summary = sess.run(summary_op, feed_dict={v_loss:G_current})
        writer.add_summary(summary, count)
        count += 1

        print("%d %d Loss=%.8f Time=%.3f (avg:%.3f)" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st, (time.time() - e_st) / cnt))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%s_00_train.jpg' % (epoch, train_id))

    # validation after each epoch
    v_start = time.time()
    losses = []
    for i in range(len(valid_ids)):
        valid_id = valid_ids[i]
        in_path = valid_input_dir + valid_id
        gt_path = valid_gt_dir + valid_id
        loss = validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image)
        losses += loss,
    summary = sess.run(summary_op, feed_dict={v_loss:np.mean(losses)})
    writer_val.add_summary(summary, count)
    print 'validation: Loss={:.8f} Time={:.3f}s'.format(np.mean(losses), time.time() - v_start)
    saver.save(sess, checkpoint_dir + 'model.ckpt')

