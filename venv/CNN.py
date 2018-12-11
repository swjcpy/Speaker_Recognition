import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],  strides=[1, 2, 2, 1], padding='SAME')

session = tf.InteractiveSession()

trainingData = np.load("trainingData.npy")
testingData = np.load("testingData.npy")
trainingLabel = np.load("trainingLabel.npy")
testingLabel = np.load("testingLabel.npy")


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,13,13,1])

Weight_conv1 = weight_variable([4,4,1,1])
h_conv1 = tf.nn.relu(conv2d(x_image,Weight_conv1))
# h_pool1 = max_pool_2(h_conv1)
h_pool1 = h_conv1

Weight_conv2 = weight_variable([4,4,1,1])
h_conv2 = tf.nn.relu(conv2d(h_pool1,Weight_conv1))
# h_pool2 = max_pool_2(h_conv2)
h_pool2 = h_conv2
Weight_full_connection3 = weight_variable([13*13,1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,13*13])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,Weight_full_connection3))

W_fc2 = weight_variable([1024,11])
y_out = tf.nn.softmax(tf.matmul(h_fc1,W_fc2))

ERM = tf.reduce_mean(tf.abs(y - y_out))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(ERM)

correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(2000):
    # batch = nextBatch(50)
    # if i%20 == 0:
    #     train_accuracy = accuracy.eval(feed_dict = {})
    print(i)
    train_step.run(feed_dict = {x:trainingData[i], y:trainingLabel[i]})

print("test accuracy %g"%accuracy.eval(feed_dict={x:testingData, y:testingLabel}))


