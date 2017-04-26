import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import struct
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os
from os.path import isfile, join
from scipy import misc
from sys import stdout
# import Image
from PIL import Image
# import cv2

# --------------------------------------------------------------------
imgDirectoryPath = 'images/lego/'
# imgTrainingDirectory = 'images/lego/training4/'
# imgTestDirectory = 'images/lego/test4/'
# subDirectory = '/diff/'
subDirectory = ''
filename = 'finalized_model.pkl'

imgTrainingDirectory = 'images/lego2/train2/'
imgTestDirectory = 'images/lego2/test2/'

n_classes = 0
nSlozek = 4
q = [0] * nSlozek

def getAllImageNamesFromFolder(image_folders):
    # Vrati pole

    X = []
    y = []

    for index, folder in enumerate(os.scandir(image_folders)):
        print(folder.name)
        pid = image_folders + folder.name   # pid = path_to_image_directory
        print(pid)
        x_subFolder = []
        iii = 0
        [x_subFolder.append(pid+'/'+file) for file in listdir(pid) if isfile(join(pid, file))]
        print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
        print(x_subFolder)
        print('------------------------------------------------------------------------')
        # X.append(x_subFolder)
        X = np.concatenate((X, x_subFolder))
        print(X)
        print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        for ii in range(len(X)):
            q[index] = 1
            y.append(q.copy())
            q[index] = 0
        # y = y + [os.path.basename(folder.name)] * len(x_subFolder)
        print(y)
    print(X)
    print(X[0])
    return X, y

def getSVMinputData(trainingDir):
    print('======================  getSVMinputData(trainingDir) ====================')
    X = np.array([[]])
    y = []
    index = 0
    for index, folder in enumerate(os.scandir(trainingDir)):
        folder = folder.name
        if (index == 0):
            nImages, X = (getMultiple_HOG(trainingDir + folder + subDirectory))
            for ii in range(nImages):
                q[index] = 1
                y.append(q.copy())
                q[index] = 0
            # y = y + [os.path.basename(folder)] * nImages
        else:
            nImages, x0 = (getMultiple_HOG(trainingDir + folder + subDirectory))
            # y = y + [os.path.basename(folder)] * nImages
            for ii in range(nImages):
                q[index] = 1
                y.append(q.copy())
                q[index] = 0
            X = np.concatenate((X, x0))

    n_classes = index + 1
    print('Loaded {} images from {} folders: '.format(len(X), index + 1))
    print('X.shape = ', X.shape)

    print('======================//  getSVMinputData(trainingDir) ====================')
    return X.copy(), y.copy()

def getMultiple_HOG(folder):
    print('================== getMultiple_HOG(folder)================')
    # h = [None]*10
    X=[]

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    print('List of imported images from {}:'.format(folder))
    for n in range(0, len(onlyfiles)):
        log = [str(n+1), '/', str(len(onlyfiles)),' ']
        print(''.join(log), join(os.path.basename(folder), onlyfiles[n]))
        im = misc.imread(join(folder, onlyfiles[n]))
        # im = misc.imread(join(folder, onlyfiles[n]), mode='L') # L = 8-bit gray
        # im = Image.open(join(folder, onlyfiles[n]))
        # im = cv2.cvtColor(cv2.imread(join(folder, onlyfiles[n])), cv2.COLOR_BGR2GRAY)
        X.append(im)

    X = np.array(X)


    # h = np.array(h)
    # print(h.shape)
    # print(h)

    # y=[0,1]
    # clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    # clf.fit(np.array(h), y)
    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(X, y)
    # print('[0.58,0.76, 1] is in: ', clf.predict([0.58, 0.76, 1]))
    nImages = X.shape[0]
    # nImages = len(X[0])

    # print('Count of images = ', nImages)
    print('nImages = ',nImages)
    # im = cv2.cvtColor(cv2.imread('images/img3.bmp'), cv2.COLOR_BGR2RGB)
    # plt.figure(0)
    # plt.imshow(im)
    # h = hog.compute(im)
    # print('hog data: ', h)
    # print('h.shape: ', len(h))
    # plt.scatter(h)

    print('================== // getMultiple_HOG(folder)================')

    return nImages, X

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
    """
    filled = '█'
    unfilled = '░'
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = filled * filledLength + unfilled * (length - filledLength)
    # print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end = "\r")
    stdout.write('\x1b[2K\r%s |%s| %s%s  %s/%s %s' % (prefix, bar, percent, '%', iteration, total, suffix))
    # Print New Line on Complete
    # if iteration == total:
    #     print()
# *********************************************************************
img_index = 59999
minst_imgs_path = 'MNIST_data/t10k-images.idx3-ubyte'
minst_labels_path = 'MNIST_data/t10k-labels.idx1-ubyte'

def read_image(file_name, fileLabels_name, idx_image):
    """
        file_name: If used for the MNIST dataset, should be either
                    train-images-idx3-ubyte or t10k-images-idx3-ubyte
        idx_image: index of the image you want to read.
    """
    img_file = open(file_name, 'r+b')
    print(img_file)
    ##########################################
    # Get basic information about the images #
    # (This is described in the webpage of 	 #
    # the database)							 #
    ##########################################
    img_file.seek(0)
    magic_number = img_file.read(4)
    magic_number = struct.unpack('>i', magic_number)
    print('Magic Number: ' + str(magic_number[0]))

    data_type = img_file.read(4)
    data_type = struct.unpack('>i', data_type)
    print('Number of Images: ' + str(data_type[0]))

    dim = img_file.read(8)
    dimr = struct.unpack('>i', dim[0:4])
    dimr = dimr[0]
    print('Number of Rows: ' + str(dimr))
    dimc = struct.unpack('>i', dim[4:])
    dimc = dimc[0]
    print('Number of Columns:' + str(dimc))

    image = np.ndarray(shape=(dimr, dimc))
    img_file.seek(16 + dimc * dimr * idx_image)

    for row in range(dimr):
        for col in range(dimc):
            tmp_d = img_file.read(1)
            tmp_d = struct.unpack('>B', tmp_d)
            image[row, col] = tmp_d[0]

    img_file.close()

    # Load everything in some numpy arrays
    with open(fileLabels_name, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)


    return image, lbl[idx_image]
# *********************************************************************

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = nSlozek
batch_size = 100

x = tf.placeholder('float', [None, 400*300*3])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([400*300*3, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

# --------------------- CNN --------------------------
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

        # 'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 128])),

        # 'W_conv4': tf.Variable(tf.random_normal([5, 5, 128, 256])),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([ 4*400*300*3, 64])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([64, n_classes]))
    }

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              # 'b_conv3': tf.Variable(tf.random_normal([128])),
              # 'b_conv4': tf.Variable(tf.random_normal([256])),
              'b_fc': tf.Variable(tf.random_normal([64])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, 300, 400, 1])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    # conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    # # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3)

    # conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
    # # Max Pooling (down-sampling)
    # conv4 = maxpool2d(conv4)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 4 * 300*400*3])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output
# --------------------- CNN --------------------------

def train_neural_network(x, XX, yy):

    prediction = convolutional_neural_network(x)
    # prediction = convolutional_neural_network(x)

    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)                   # with hm_epochs = 10 => Accuracy: 0.9523
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)    # with hm_epochs = 10 => Accuracy: 0.9541
                                                         # with hm_epochs = 10; batch_size = 10 => Accuracy: 0.9684

    # hm_epochs = 1       #hm_epochs = 1 => Accuracy: 0.9194
    hm_epochs = 15

    # model_saver = tf.train.Saver()

    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            start_time = time.time()
            epoch_loss = 0
            # for _ in range(int(mnist.train.num_examples / batch_size)):
            for ii in range(XX.shape[0]):
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = [np.array(XX[ii].flatten(), dtype=float)]
                epoch_y = yy[ii]
                # print(epoch_x)
                # print(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # todo: mereni casu pro kazdou epochu
            end_time = time.time()
            # print("total epoch time: ", round(end_time - start_time), "s")
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'for time', round(end_time - start_time), 's, loss:', epoch_loss)


        # construct model
        # pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax


        # # --------------------- Save and restore trained  model ---------------
        # model_saver.save(sess, "saved_models/CNN_New-MNIST.ckpt")
        #
        # # Restore the model
        # with tf.Session() as session2:
        #     model_saver.restore(session2, "saved_models/CNN_New.ckpt")
        #     print("Model restored.")
        #     print('Initialized')
        #
        #     # Check Variable
        #     # W1 = session2.run(W1)
        #     # print(W1)
        # # ------------------- Save and restore trained  model ---------------

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


        X_test, y_test = getSVMinputData(imgTestDirectory)
        # y_test = [['1','0'],['1','0'],['1','0'],['0','1'],['0','1'],['0','1']]

        XXX = []
        print('.....................................................')
        for iii in range(X_test.shape[0]):
            XXX.append(np.array(X_test[iii].flatten(), dtype=float))

            # ....................................................
            x_in = [np.array(X_test[iii].flatten(), dtype=float)]
            # print(y_test[iii], ' => digit: ', np.argmax(y_test[iii]))
            classification = sess.run(tf.argmax(prediction, 1), feed_dict={x: x_in})

            # print("correct digit = ", np.argmax(y_test[iii]), ";         classification = ", classification[0])
            print('Neural Network predicted', classification, "for LEGO piece", np.argmax(y_test[iii]))

        print('.....................................................')
        # X_test = [np.array(X_test[0].flatten(), dtype=float)]
        print('Accuracy:', accuracy.eval({x: XXX, y: y_test}))




        # =======================================================================
        # testIndex = 0   # index 0=>7; 1=>2 ; 2=>1 ; 3=>0; 4=>4
        # x_in = np.expand_dims(mnist.test.images[testIndex], axis=0)
        # y_in = np.expand_dims(mnist.test.labels[testIndex], axis=0)
        # print(y_in[0],' => digit: ', np.argmax(y_in[0]))
        #
        # classification = sess.run(tf.argmax(prediction, 1), feed_dict={x: x_in})
        #
        # print("correct digit = ", np.argmax(y_in[0]), "; classification = ", classification[0])
        # print('Neural Network predicted', classification, "for your digit")
        # -----------------------------------------------------------------------
        """
        yy =[[1,0,0,0,0,0,0,0,0,0]]
        yy1=[[0,1,0,0,0,0,0,0,0,0]]
        yy2=[[0,0,1,0,0,0,0,0,0,0]]
        yy3=[[0,0,0,1,0,0,0,0,0,0]]
        yy4=[[0,0,0,0,1,0,0,0,0,0]]
        yy5=[[0,0,0,0,0,1,0,0,0,0]]
        yy6=[[0,0,0,0,0,0,1,0,0,0]]
        yy7=[[0,0,0,0,0,0,0,1,0,0]]
        yy8=[[0,0,0,0,0,0,0,0,1,0]]
        yy9=[[0,0,0,0,0,0,0,0,0,1]]
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy1}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy2}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy3}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy4}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy5}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy6}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy7}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy8}))      # we will get either 0% or 100%.
        print("Accuracy image:", accuracy.eval({x: x_in, y: yy9}))      # we will get either 0% or 100%.
        """
        # =======================================================================
        # read image as bitmap
        # image, label = read_image(minst_imgs_path, minst_labels_path, testIndex)
        # img_plot = plt.title('Label = ' + str(label))
        # img_plot = plt.imshow(image, 'Greys')  # ; plt.title('Label = ' + str(label))
        # # image = tf.placeholder(tf.float32, shape=[None, 9, None])
        # # image = tf.reshape(image, tf.stack([28, 28, 1]))
        # # image = tf.reshape(image, [28, 28, 1])
        # # flat_inputs = tf.contrib.layers.flatten(image)
        # image = np.array(image.flatten(), dtype=float)
        # image = [image]
        # # print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')
        # # print(x_in)
        # # print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        # # # print(image.flatten())
        # # print(image)
        # # # np.array(image.flatten(), dtype=float)
        # # # print(flat_inputs)
        # # print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')
        # print('classification the same image as a bitmap:')
        # classification = sess.run(tf.argmax(prediction, 1), feed_dict={x: image})
        # print("correct digit = ", label, "; classification = ", classification[0])

        # image = tf.image.decode_jpeg(image, channels=1)
        # image = tf.cast(image, tf.float)
        # print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
        # classification = sess.run(tf.argmax(prediction, 1), feed_dict={x: image})
        # print("correct digit = ", label, "; classification = ", classification)

        # with tf.Session() as sess:
        #     # Feed the image_data as input to the graph and get first prediction
        #     # softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        #
        #     predictions = sess.run({'DecodeJpeg/contents:0': x_in})
        #
        #     # Sort to show labels of first prediction in order of confidence
        #     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #
        #     for node_id in top_k:
        #         human_string = yy[node_id]
        #         score = predictions[0][node_id]
        #         print('%s (score = %.5f)' % (human_string, score))
        #
        # with tf.Session() as sess:
        #     coord = tf.train.Coordinator()
        #
        #     threads = tf.train.start_queue_runners(coord=coord)
        #
        #     res = sess.run(x_in)
        #
        #     coord.request_stop()
        #
        #     coord.join(threads)
        # ----------------------------------

        # # Read in the image_data
        # image_data = tf.gfile.FastGFile(image_path, 'rb').read()



        # ----------------------------------

        # image, label = read_image(minst_imgs_path, minst_labels_path, 0)

        # print('Accuracy 2:', accuracy.eval({x: image, y: label}))
        # print('Accuracy 3:', accuracy.eval(image, 5))

        # img_plot = plt.title('Label = ' + str(label))
        # img_plot = plt.imshow(image, 'Greys')  # ; plt.title('Label = ' + str(label))
        print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')

def train_neural_network2(x, XX, yy):

    prediction = convolutional_neural_network(x)
    # prediction = convolutional_neural_network(x)

    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # with hm_epochs = 10 => Accuracy: 0.9523
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)    # with hm_epochs = 10 => Accuracy: 0.9541
    # with hm_epochs = 10; batch_size = 10 => Accuracy: 0.9684

    # hm_epochs = 1       #hm_epochs = 1 => Accuracy: 0.9194
    hm_epochs = 2

    # model_saver = tf.train.Saver()

    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            start_time = time.time()
            epoch_loss = 0
            # for _ in range(int(mnist.train.num_examples / batch_size)):
            for ii in range(XX.shape[0]):
                printProgressBar(ii, len(XX), prefix='Progress:', suffix='Complete', length=50)

                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                image = misc.imread(XX[ii])

                epoch_x = [np.array(image.flatten(), dtype=float)]
                epoch_y = yy[ii]
                # print(epoch_x)
                # print(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # todo: mereni casu pro kazdou epochu
            end_time = time.time()
            # print("total epoch time: ", round(end_time - start_time), "s")
            # \r vymaze(prepise) predesli radek v konzoli, v tomto pripade ProgressBar a nahradiho informacnim vypisem
            print('\rEpoch', epoch + 1, 'completed out of', hm_epochs, 'for time', round(end_time - start_time),
                  's, loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        X_test, y_test = getSVMinputData(imgTestDirectory)
        # y_test = [['1','0'],['1','0'],['1','0'],['0','1'],['0','1'],['0','1']]

        XXX = []
        print('.....................................................')
        for iii in range(X_test.shape[0]):
            XXX.append(np.array(X_test[iii].flatten(), dtype=float))

            # ....................................................
            x_in = [np.array(X_test[iii].flatten(), dtype=float)]
            # print(y_test[iii], ' => digit: ', np.argmax(y_test[iii]))
            classification = sess.run(tf.argmax(prediction, 1), feed_dict={x: x_in})

            # print("correct digit = ", np.argmax(y_test[iii]), ";         classification = ", classification[0])
            print('Neural Network predicted', classification, "for LEGO piece", np.argmax(y_test[iii]))

        print('.....................................................')
        # X_test = [np.array(X_test[0].flatten(), dtype=float)]
        print('Accuracy:', accuracy.eval({x: XXX, y: y_test}))
        print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# XX, yy = getSVMinputData(imgTrainingDirectory)
# print('yy ', yy)  # yy = [['1', '0'], ['1', '0'], ['1', '0'], ['0', '1'], ['0', '1'], ['0', '1']]

# train_neural_network(x, XX, yy) #   , XX.shape[0]
# //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

XX, yy = getAllImageNamesFromFolder(imgTrainingDirectory)
print('yy ', yy)
train_neural_network2(x, XX, yy) #   , XX.shape[0]

"""
X, y = getSVMinputData(imgTrainingDirectory)

print(X[0].shape)
# image = misc.imread('./images/img3.bmp')
plt.figure(0)
plt.title('Label = ' + str(y[0]))
plt.imshow(X[0], 'Greys')  # ; plt.title('Label = ' + str(label))

plt.figure(1)
img_plot2 = plt.title('Label = ' + str(y[22]))
img_plot2 = plt.imshow(X[22], 'Greys')  # ; plt.title('Label = ' + str(label))

plt.figure(2)
img_plot3 = plt.title('Label = ' + str(y[40]))
img_plot3 = plt.imshow(X[40], 'Greys')  # ; plt.title('Label = ' + str(label))
"""





plt.show()