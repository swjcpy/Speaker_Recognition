import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], padding='SAME')

def avg_pool_2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 13, 13, 1])
y = tf.placeholder(tf.float32, [None, 10])

Weight_conv1 = weight_variable([4,4,1,1])
h_conv1 = tf.nn.relu(conv2d(x,Weight_conv1))
h_pool1 = max_pool_2(h_conv1)

Weight_conv2 = weight_variable([4,4,1,1])
h_conv2 = tf.nn.relu(conv2d(h_pool1,Weight_conv1))
h_pool2 = max_pool_2(h_conv2)

Weight_full_connection3 = weight_variable([9,1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,9])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,Weight_full_connection3))

ERM = tf.reduce_mean(tf.abs(y - h_fc1))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(ERM)

correct_prediction =

