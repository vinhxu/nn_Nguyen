# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_data import *




img_size_flat = 2  # 28x28=784, the total number of pixels
n_classes = 2     # Number of classes, one class per digit


def print_data(data_array, comment =""):
    print comment
    for i in data_array:
        print i

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode="train")
x_test, y_test = load_data(mode='test')

# print(len(x_test))
# print("Size of:")
# print("- Training-set:\t\t{}".format(len(y_train)))
# print("- Validation-set:\t{}".format(len(y_valid)))
# print('x_train:\t{}'.format(x_train.shape))
# print('y_train:\t{}'.format(y_train.shape))
# print('x_valid:\t{}'.format(x_valid.shape))
# print('y_valid:\t{}'.format(y_valid.shape))

# Size of:
# - Training-set:		1558
# - Validation-set:	137
# x_train:	(1558, 2)
# y_train:	(1558, 2)
# x_train:	(137, 2)
# y_valid:	(137, 2)

# print (x_train[:5], "1st 5 data in x_train")
# print (y_train[:5], "1st 5 data in y_train")
# print (x_valid[:5], "1st 5 data in x_valid")
# print (y_valid[:5], "1st 5 data in y_valid")
# print (x_test[:5], "1st 5 data in x_test")
# print (y_test[:5], "1st 5 data in y_test")


# (array([[1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.]]), '1st 5 data in y_train')
# (array([[1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.]]), '1st 5 data in y_valid')
# (array([[1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.],
#        [1., 0.]]), '1st 5 data in y_test')

# Hyper-parameters
epochs = 100       # Total number of training epochs
batch_size = 50    # Training batch size
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.01   # The optimization initial learning rate

h1 = 1               # Number of units in the first hidden layer
h2 = 1
h3 = 1

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)

def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')


fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
fc2 = fc_layer(fc1, h2, 'FC2', use_relu=True)
fc3 = fc_layer(fc2, h3, 'FC3', use_relu=True)
output_logits = fc_layer(fc3, n_classes, 'OUT', use_relu=True)
# print ('output_logits: ' + str(output_logits))

# Network predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
y_true_test  = tf.argmax(y_test, axis=1, name='y_true_test')
# print ('cls_prediction: ' + str(cls_prediction))

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Create the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()

def train_nn_model(x_train, y_train, saveModelBoolean = True):
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                 feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid[:], y: y_valid[:]}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')
        if(acc_valid >=0.99):
            modelName = "/Users/vinhxu/Desktop/Bachelor thesis/03_model_saved/.1nn_validation_Loss{0:.2f}_Accuracy{1:.01%}.ckpt".format(loss_valid, acc_valid)
            save_path = saver.save(sess, modelName)
            break

    if (saveModelBoolean):
        modelName = "./.1nn_validation_Loss{0:.2f}_Accuracy{1:.01%}.ckpt".format(loss_valid, acc_valid)
        save_path = saver.save(sess, modelName)


def plot_images(images, cls_true, cls_pred=None, title=None, plot=plt, color='b'):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    ax = plt.gca()
    ax.plot(images[:,0], images[:,1], marker="o", color=color, ls="", ms=.7)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the images.
    plot_images(images=incorrect_images[:],
                cls_true=cls_true[:],
                cls_pred=cls_pred[:],
                title=title,color='red')


def plot_example_correct(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    correct = np.equal(cls_pred, cls_true).astype(bool)

    # Get the images from the test-set that have been
    # incorrectly classified.
    correct_images = images[correct]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[correct]
    cls_true = cls_true[correct]

    # Plot the images.
    plot_images(images=correct_images[:],
                cls_true=cls_true[:],
                cls_pred=cls_pred[:],
                title=title,color='green')


def get_test_date(dataName="test"):
    if (dataName == "train"):
        return {x: x_train[:], y: y_train[:]}
    elif (dataName == "validation"):
        return {x: x_valid[:], y: y_valid[:]}
    else:
        return {x: x_test[:], y: y_test[:]}

def test_nn_model(modelName="/Users/vinhxu/Desktop/Bachelor thesis/03_model_saved/.1nn.ckpt", dataName = "test"):
    # Restore variables from disk.
    saver.restore(sess, modelName)

    # Get test data
    feed_dict_test = get_test_date(dataName)
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')

    # Plot some of the correct and misclassified examples
    plt.figure()
    output = sess.run(output_logits, feed_dict=feed_dict_test)
    cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
    cls_true = np.argmax(feed_dict_test.get(y), axis=1)
    plot_images(feed_dict_test.get(x), cls_true, cls_pred, title='Correct Examples', color='green')
    plot_example_errors(feed_dict_test.get(x), cls_true, cls_pred, title='Misclassified Examples')
    plt.show()

    for i, c in enumerate(cls_true):
        print "output: " + str(output[i]) + "  , cls_pred: " + str(cls_pred[i]) + "  , correct: " + str(c)
    print(len(cls_true))

# train_nn_model(x_train, y_train)
# test_nn_model(dataName="validation")
test_nn_model(modelName="./.1nn_validation_Loss0.27_Accuracy92.7%.ckpt", dataName="train")
# test_nn_model(dataName="train")


