import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tqdm
from functools import reduce

ROWS = 28
COLS = 28

def generator(inputs, nch=200):
    # inputs is [?, 100] 
    H = inputs
    with tf.variable_scope("dense"):
        fanin = H.get_shape().as_list()[1]
        fanout = nch * (ROWS // 2) * (COLS // 2)
        with tf.device("/cpu:0"):
            W = tf.get_variable("weights", [fanin, fanout], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.matmul(H, W) + b
        with tf.device("/cpu:0"):
            mean, variance = tf.nn.moments(activation, [1])
        normalized = (activation - tf.expand_dims(mean, 1)) / tf.expand_dims(tf.sqrt(variance), 1)
        dense1 = tf.nn.relu(normalized)
    H = dense1
    with tf.name_scope("reshaping"):
        reshaped = tf.reshape(H, [tf.shape(inputs)[0], ROWS // 2, COLS // 2, nch])
        upsampled = tf.image.resize_images(reshaped, [ROWS, COLS])
    H = upsampled
    with tf.variable_scope("conv1"):
        filter_size = (3, 3)
        fanin = H.get_shape().as_list()[-1]
        fanout = nch // 2
        with tf.device("/cpu:0"):
            W = tf.get_variable("filter", [filter_size[0], filter_size[1], fanin, fanout],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.nn.bias_add(tf.nn.conv2d(H, W, [1, 1, 1, 1], "SAME"), b)
        with tf.device("/cpu:0"):
            mean, variance = tf.nn.moments(activation, [1])
        normalized = (activation - tf.expand_dims(mean, 1)) / tf.expand_dims(tf.sqrt(variance), 1)
        conv1 = tf.nn.relu(normalized)
    H = conv1
    with tf.variable_scope("conv2"):
        filter_size = (3, 3)
        fanin = H.get_shape().as_list()[-1]
        fanout = nch // 3
        with tf.device("/cpu:0"):
            W = tf.get_variable("filter", [filter_size[0], filter_size[1], fanin, fanout],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.nn.bias_add(tf.nn.conv2d(H, W, [1, 1, 1, 1], "SAME"), b)
        with tf.device("/cpu:0"):
            mean, variance = tf.nn.moments(activation, [1])
        normalized = (activation - tf.expand_dims(mean, 1)) / tf.expand_dims(tf.sqrt(variance), 1)
        conv2 = tf.nn.relu(normalized)
    H = conv2
    with tf.variable_scope("conv3"):
        filter_size = (1, 1)
        fanin = H.get_shape().as_list()[-1]
        fanout = 1
        with tf.device("/cpu:0"):
            W = tf.get_variable("filter", [filter_size[0], filter_size[1], fanin, fanout],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.nn.bias_add(tf.nn.conv2d(H, W, [1, 1, 1, 1], "SAME"), b)
        conv3 = tf.nn.sigmoid(activation)
    return conv3

def discriminator(inputs, dropout=0.25):
    # inputs is [?, 28, 28, 1] 
    H = inputs
    with tf.variable_scope("conv1"):
        filter_size = (5, 5)
        fanin = H.get_shape().as_list()[-1]
        fanout = 256
        with tf.device("/cpu:0"):
            W = tf.get_variable("filter", [filter_size[0], filter_size[1], fanin, fanout],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.nn.bias_add(tf.nn.conv2d(H, W, [1, 2, 2, 1], "SAME"), b)
        conv1 = tf.nn.dropout(tf.maximum(0.2 * activation, activation), 1.0 - dropout)
    H = conv1
    with tf.variable_scope("conv2"):
        filter_size = (5, 5)
        fanin = H.get_shape().as_list()[-1]
        fanout = 512
        with tf.device("/cpu:0"):
            W = tf.get_variable("filter", [filter_size[0], filter_size[1], fanin, fanout],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.nn.bias_add(tf.nn.conv2d(H, W, [1, 2, 2, 1], "SAME"), b)
        conv2 = tf.nn.dropout(tf.maximum(0.2 * activation, activation), 1.0 - dropout)
    H = conv2
    with tf.name_scope("flatten"):
        flattened = tf.reshape(H, [tf.shape(H)[0], reduce(lambda x, y: x * y, H.get_shape().as_list()[1:], 1)])
    H = flattened
    with tf.variable_scope("dense1"):
        fanin = H.get_shape().as_list()[1]
        fanout = 256 
        with tf.device("/cpu:0"):
            W = tf.get_variable("weights", [fanin, fanout], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        activation = tf.matmul(H, W) + b
        dense1 = tf.nn.dropout(tf.maximum(0.2 * activation, activation), 1.0 - dropout)
    H = dense1
    with tf.variable_scope("softmax"):
        fanin = H.get_shape().as_list()[1]
        fanout = 2 
        with tf.device("/cpu:0"):
            W = tf.get_variable("weights", [fanin, fanout], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(mean=0.0,
                        stddev=1.0 / (fanin * fanout), dtype=tf.float32))
            b = tf.get_variable("bias", [fanout], dtype=tf.float32,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))
        logits = tf.matmul(H, W) + b
        softmax = tf.nn.softmax(logits)
        pred = tf.argmax(logits, 1)
    return logits, softmax, pred

def graph(use_gpu=False):
    gpu = "/gpu:0" if use_gpu else "/cpu:0"
    with tf.device(gpu):
        with tf.name_scope("gen"):
            seed = tf.placeholder(tf.float32, [None, 100], name="noise_seed")
            with tf.variable_scope("generator"):
                generated_images = generator(seed)
            with tf.name_scope("labels"):
                false_labels_dis = tf.one_hot(tf.fill([tf.shape(generated_images)[0]], 0), 2, on_value=1.0, off_value=0.0)
                false_labels_gen = tf.one_hot(tf.fill([tf.shape(generated_images)[0]], 1), 2, on_value=1.0, off_value=0.0)
            with tf.device("/cpu:0"):
                tf.summary.image("generated", generated_images) 
        with tf.variable_scope("mnist_images"):
            with tf.name_scope("input"):
                mnist_inputs_flat = tf.placeholder(tf.float32, [None, 784])
                mnist_inputs = tf.reshape(mnist_inputs_flat, [tf.shape(mnist_inputs_flat)[0], 28, 28, 1])
                with tf.device("/cpu:0"):
                    tf.summary.image("actual", mnist_inputs) 
            with tf.name_scope("labels"):
                true_labels = tf.one_hot(tf.fill([tf.shape(mnist_inputs)[0]], 1), 2, on_value=1.0, off_value=0.0)
        with tf.name_scope("dis"):
            with tf.variable_scope("discriminator"):
                mnist_logits, mnist_proba, mnist_prediction = discriminator(mnist_inputs)
            with tf.variable_scope("discriminator", reuse=True):
                generated_logits, generated_proba, generated_prediction = discriminator(generated_images)
        with tf.name_scope("loss"):
            mnist_cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mnist_logits, true_labels))
            generated_cross_entropy_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(generated_logits, false_labels_gen))
            generated_cross_entropy_loss_dis = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(generated_logits, false_labels_dis))
            cross_entropy_loss_dis = 0.5 * (generated_cross_entropy_loss_dis + mnist_cross_entropy_loss)
            cross_entropy_loss_gen = generated_cross_entropy_loss_gen
            with tf.device("/cpu:0"):
                tf.summary.scalar("dis-loss", cross_entropy_loss_dis)
                tf.summary.scalar("gen-loss", cross_entropy_loss_gen)
            true_positive_ratio = tf.reduce_mean(tf.cast(tf.equal(mnist_prediction, 1), tf.float32))
            false_positive_ratio = tf.reduce_mean(tf.cast(tf.equal(generated_prediction, 1), tf.float32))
            with tf.device("/cpu:0"):
                tf.summary.scalar("false-positive-rate", false_positive_ratio);
                tf.summary.scalar("true-positive-rate", true_positive_ratio);
    with tf.device("/cpu:0"):
        with tf.name_scope("training_dis"):
            dis_opt = tf.train.AdamOptimizer(1e-4)
            with tf.device(gpu):
                dis_grads = dis_opt.compute_gradients(cross_entropy_loss_dis, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))
            train_dis = dis_opt.apply_gradients(dis_grads)
        with tf.name_scope("training_gen"):
            gen_opt = tf.train.AdamOptimizer(1e-3)
            with tf.device(gpu):
                gen_grads = gen_opt.compute_gradients(cross_entropy_loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))
            train_gen = gen_opt.apply_gradients(gen_grads)
    return train_dis, train_gen, seed, mnist_inputs_flat

def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_dis, train_gen, seed_input, mnist_input = graph(use_gpu=True)
    summaries = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ONE_GEN_FOR_EVERY_X_DIS = 1
    BURNIN = 10
    ITERATIONS = 10000
    BATCH_SIZE = 100
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard
        sess.run(init)
        for i in tqdm.tqdm(range(ITERATIONS)):
            mnist_inputs, _ = mnist.train.next_batch(BATCH_SIZE // 2)
            seed_inputs = np.random.normal(size=[BATCH_SIZE // 2, 100])
            seed_inputs[seed_inputs > 1] = 1.0
            seed_inputs[seed_inputs < -1] = -1.0
            feed_dict = {
                seed_input: seed_inputs,
                mnist_input: mnist_inputs
            }
            smry = sess.run([train_dis, summaries], feed_dict=feed_dict)[1]
            if i >= BURNIN: # and (i + 1) % ONE_GEN_FOR_EVERY_X_DIS == 0:
                smry = sess.run([train_gen, summaries], feed_dict=feed_dict)[1]
            summary_writer.add_summary(smry, global_step=i)
            summary_writer.flush()

train()
