import os
import numpy as np
import cv2
import tensorflow as tf
import time


def corrupt_image(img_path, mask_path):
    image = cv2.imread(img_path)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY).astype(np.uint8)
    corrupt_img = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("", corrupt_img)
    cv2.waitKey()
    cv2.imwrite(corrupt_img_path, corrupt_img)


def Sequential_1(input_tensor, max_iter, iter_i=0):
    print("s_1___{}".format(iter_i))

    down_sample_tensor = tf.contrib.layers.conv2d(inputs=input_tensor,#1
                                           num_outputs=128,
                                           kernel_size=3,
                                           stride=2,
                                           padding='SAME',
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           activation_fn=tf.nn.leaky_relu)#3

    blurred_tensor = tf.contrib.layers.conv2d(inputs=down_sample_tensor,#4
                                           num_outputs=128,
                                           kernel_size=3,
                                           stride=1,
                                           padding='SAME',
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           activation_fn=tf.nn.leaky_relu)#6
    if iter_i < max_iter:
        blurred_tensor = Sequential_7(blurred_tensor, max_iter, iter_i + 1)
    print("upsample_{}".format(iter_i), np.array(blurred_tensor.shape))
    output_tensor = tf.image.resize_images(blurred_tensor, tf.shape(blurred_tensor)[1:3] * 2, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output_tensor

def Sequential_7(input_tensor, max_iter, iter_i=0):
    Sequential_0 = tf.contrib.layers.conv2d(inputs=input_tensor,  # 1
                                            num_outputs=128,
                                            kernel_size=1,
                                            stride=1,
                                            padding='SAME',
                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                            activation_fn=tf.nn.leaky_relu)  # 3
    skip_tensor = tf.concat([Sequential_0, Sequential_1(input_tensor, max_iter, iter_i)], 3)#1

    print(skip_tensor)
    batch_norm_tensor = tf.contrib.layers.batch_norm(inputs=skip_tensor, activation_fn=None)  # 2
    blurred_tensor = tf.contrib.layers.conv2d(inputs=batch_norm_tensor,
                                              num_outputs=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding='SAME',
                                              normalizer_fn=tf.contrib.layers.batch_norm,
                                              activation_fn=tf.nn.leaky_relu)  # 5
    output_tensor = tf.contrib.layers.conv2d(inputs=blurred_tensor,
                                             num_outputs=128,
                                             kernel_size=1,
                                             stride=1,
                                             padding='SAME',
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             activation_fn=tf.nn.leaky_relu)  # 8
    return output_tensor

def skip(input_tensor,
         len_depth,
         num_output_channels=3):
    end_points = {}
    end_points["input_tensor"] = input_tensor
    Sequential_7_tensor = Sequential_7(input_tensor, len_depth)
    skip_tensor = tf.contrib.layers.conv2d(inputs=Sequential_7_tensor,
                                           num_outputs=num_output_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding='SAME',
                                           normalizer_fn=None,
                                           activation_fn=tf.nn.sigmoid)  # 10
    return skip_tensor, end_points


lastTime = time.time()
os.makedirs("./output", exist_ok=True)
### setup
num_iter = 3001
net_depth = 4
show_every = 50
reg_noise_std = 0.00
param_noise = True

LR = 0.01

depth = np.power(2, net_depth + 1)


corrupt_img_path = './data/inpainting/corrupt_library.png'
image_name = corrupt_img_path.rsplit("/", 1)[1].rsplit(".", 1)[0]


image = cv2.imread(corrupt_img_path)

dim_0 = image.shape[0]
dim_1 = image.shape[1]

new_dim = min(dim_0 - dim_0 % depth, dim_1 - dim_1 % depth)
max_image = image.max()



image_tensor = tf.constant(image / max_image, dtype=tf.float32)
mask_tensor = tf.constant(image > 0, dtype=tf.float32)

input_tensor = tf.expand_dims(tf.image.resize_images(image_tensor, (new_dim, new_dim), tf.image.ResizeMethod.BICUBIC), 0)


net_input_tensor = tf.placeholder(tf.float32, input_tensor.shape)
net_deeper_tensor = tf.contrib.layers.conv2d(
        inputs=net_input_tensor,
        num_outputs=depth,
        kernel_size=1,
        stride=1,
        padding='SAME',
        normalizer_fn=None,
        activation_fn=None)
skip_tensor, end_points = skip(net_deeper_tensor, net_depth)
out_tensor = tf.image.resize_images(tf.squeeze(skip_tensor, axis=0), (dim_0, dim_1), tf.image.ResizeMethod.BICUBIC)

#print([x for x in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if len(x.shape) == 4])


# Compute number of parameters#
s = sum(np.prod(list(p.shape)) for p in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
print('Number of params: %d' % s)

if param_noise:
    for n in [x for x in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if len(x.shape) == 4]:
        up_op = tf.assign(n, n + tf.random_normal(tf.shape(n)) * tf.reduce_mean(tf.square(n - tf.reduce_mean(n))) / 50)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, up_op)

net_input_saved = tf.identity(net_input_tensor)
noise = tf.identity(net_input_tensor)
if reg_noise_std > 0:
    net_input_tensor = net_input_saved + (tf.random_normal(tf.shape(noise)) * reg_noise_std)


total_loss = tf.losses.mean_squared_error(image_tensor * mask_tensor, out_tensor * mask_tensor)


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(LR).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_image = sess.run(image_tensor) * max_image
    cv2.imwrite("./output/corrupt.png", out_image)
    for ii in range(num_iter):
        net_input = np.random.uniform(0, 0.1, size=(1, new_dim, new_dim, 3))
        sess.run(train_op, feed_dict={net_input_tensor: net_input})
        if ii % show_every == 0:
            out_data, loss = sess.run([out_tensor, total_loss], feed_dict={net_input_tensor: net_input})
            print("step: {} loss: {}\tuse {} secs".format(ii, loss, time.time() - lastTime))
            lastTime = time.time()
            out_image = out_data * max_image
            cv2.imwrite("./output/{}_{}.png".format(ii, image_name), out_image)



