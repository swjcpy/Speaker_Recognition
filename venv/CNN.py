import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],  strides=[1, 2, 2, 1], padding='SAME')

def nextBatch(n):
    listX = []
    listY = []
    data = np.load('trainingData.npy')
    label = np.load('trainingLabel.npy')
    length = len(data)
    num = [i for i in range(length)]
    np.random.shuffle(num)
    for i in range(n):
        listX.append(data[num[i]])
        listY.append(label[num[i]])
    return listX,listY

session = tf.InteractiveSession()
listTestAcc = []
listTrainAcc = []
trainingData = np.load("trainingData.npy")
testingData = np.load("testingData.npy")
trainingLabel = np.load("trainingLabel.npy")
testingLabel = np.load("testingLabel.npy")


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# print(x.shape)
x_image = tf.reshape(x,[-1,16,16,1])

Weight_conv1 = weight_variable([5,5,1,2])
b_conv1 = bias_variable([2])
h_conv1 = tf.nn.relu(conv2d(x_image,Weight_conv1)+ b_conv1)
h_pool1 = max_pool_2(h_conv1)
# h_pool1 = h_conv1

Weight_conv2 = weight_variable([5,5,2,4])
b_conv2 = bias_variable(([4]))
h_conv2 = tf.nn.relu(conv2d(h_pool1,Weight_conv2)+b_conv2)
h_pool2 = max_pool_2(h_conv2)
# h_pool2 = h_conv2

Weight_full_connection3 = weight_variable([4*4*4,1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,4*4*4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,Weight_full_connection3))

W_fc2 = weight_variable([1024,11])
y_out = tf.nn.softmax(tf.matmul(h_fc1,W_fc2))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batchX, batchY = nextBatch(50)
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batchX, y: batchY})
        test_accuracy = accuracy.eval(feed_dict={x:testingData, y:testingLabel})
        print("test data ",i,": ",test_accuracy)
        print("train data ",i,": ",train_accuracy)
        listTestAcc.append(test_accuracy)
        listTrainAcc.append(train_accuracy)
    if i%100 == 0:
        np.save('BatchedTestAcc',listTestAcc)
        np.save('BatchedTrainAcc',listTrainAcc)
        print("saved")
    # print(i)

    train_step.run(feed_dict = {x:batchX, y:batchY})


print("test accuracy %g"%accuracy.eval(feed_dict={x:testingData, y:testingLabel}))
np.save('BatchedTestAcc',listTestAcc)
np.save('BatchedTrainAcc',listTrainAcc)


