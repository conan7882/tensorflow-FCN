import numpy as np
import argparse
import sys, os
import scipy.io
import data_image 

import tensorflow as tf 

class DeepContour(object):
    def __init__(self, num_classes, num_channels, skip_layer, is_training, weights_path = 'DEFAULT', is_FCN = False):
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - num_classes: int, number of classes of the new dataset
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights,
                    (if bvlc_alexnet.npy is not in the same folder)
        """
        self.NUM_CLASSES = num_classes
        # self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.IS_TRAINING = is_training
        self.IS_FCN = is_FCN
        self.NUM_CHANNELS = num_channels

        self.X = tf.placeholder(tf.float32, [None, None, None, num_channels])
        self.KEEP_PROB = tf.placeholder(tf.float32)
        if is_training:
            self.LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
            self.Y_ = tf.placeholder(tf.int64, [None, None, None])
            self.MASK_ = tf.placeholder(tf.int32, [None, None, None])

    
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = ''
        else:
            self.WEIGHTS_PATH = weights_path

        if self.IS_FCN:
            self.create_FCN()
        else:
            self.create()


        if is_training:
            mask_ = self.MASK_
            y_ = self.Y_
            with tf.name_scope('loss'):
                # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                #     (logits = tf.boolean_mask(self.dconv3, mask_),labels = tf.boolean_mask(y_, mask_)))
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = apply_mask(self.dconv3, mask_),labels = apply_mask(y_, mask_)))
            tf.summary.scalar('loss', loss)
                
            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
                grads = optimizer.compute_gradients(loss)
                self.train_step = optimizer.apply_gradients(grads)
                # self.train_step = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(loss)
            [tf.summary.histogram('gradient/' + var.name, grad) for grad, var in grads]
            

            with tf.name_scope('accuracy'):
                # correct_prediction = tf.boolean_mask(tf.equal(self.prediction, y_), mask_)
                correct_prediction = apply_mask(tf.equal(self.prediction, y_), mask_)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.image("Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1))

            self.merged_summary = tf.summary.merge_all()


    def create_FCN(self):
        conv1 = conv(self.X, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, name = 'pool1')

        conv2 = conv(pool1, 3, 3, 48, 'conv2')
        pool2 = max_pool(conv2, name = 'pool2')

        conv3 = conv(pool2, 3, 3, 64, 'conv3')
        pool3 = max_pool(conv3, name = 'pool3')

        conv4 = conv(pool3, 3, 3, 128, 'conv4')
        pool4 = max_pool(conv4, name = 'pool4')
        # h_pool4_flat = tf.reshape(pool4, [-1, 2*2*128, 1, 1])

        fc1 = conv(pool4, 2, 2, 128, 'fc1', padding = 'SAME')
        dropout_fc1 = dropout(fc1, self.KEEP_PROB)

        self.fc2 = conv(dropout_fc1, 1, 1, self.NUM_CLASSES, 'fc2', padding = 'SAME', relu = False)

        # deconv
        dconv1 = dconv(self.fc2, pool3, 4, 4, 'dconv1')
        dconv2 = dconv(dconv1, pool2, 4, 4, 'dconv2')

        shape_X = tf.shape(self.X)
        deconv3_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], self.NUM_CLASSES])
        self.dconv3 = dconv(dconv2, None, 16, 16, 'dconv3', 
            output_channels = self.NUM_CLASSES, output_shape = deconv3_shape, stride_x = 4, stride_y = 4)
        self.prediction = tf.argmax(self.dconv3, name="prediction", dimension = -1)

    def create(self):
        conv1 = conv(self.X, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, name = 'pool1')

        conv2 = conv(pool1, 3, 3, 48, 'conv2')
        pool2 = max_pool(conv2, name = 'pool2')

        conv3 = conv(pool2, 3, 3, 64, 'conv3')
        pool3 = max_pool(conv3, name = 'pool3')

        conv4 = conv(pool3, 3, 3, 128, 'conv4')
        pool4 = max_pool(conv4, name = 'pool4')
        h_pool4_flat = tf.reshape(pool4, [-1, 2*2*128])

        self.fc_feature, fc1 = fc(h_pool4_flat, 2*2*128, 128, 'fc1')
        dropout_fc1 = dropout(fc1, self.KEEP_PROB)

        self.fc2 = fc(dropout_fc1, 128, self.NUM_CLASSES, 'fc2', relu = False)[0]



    def restore_trained_parameters(self, session, saver, model_path, load_name):
        print('Restoring trained parameters...')
        print(model_path + load_name)
        saver.restore(session, model_path + load_name)
        print('Restore all trained parameters done!')


    def load_trained_parameters(self, session):
        print('Loading trained parameters...')
        if self.IS_FCN:
            weights_dict = np.load(self.WEIGHTS_PATH + 'DeepContourDeepFeature_FCN.npy').item()
            print('DeepContourDeepFeature_FCN.npy')
        else:
            weights_dict = np.load(self.WEIGHTS_PATH + 'DeepContourDeepFeature.npy').item()
            print('DeepContourDeepFeature.npy')
        for layer_name in weights_dict:
            
            with tf.variable_scope(layer_name, reuse = True):
                for data in weights_dict[layer_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable = False)
                    else:
                        var = tf.get_variable('weights', trainable = False)
                    session.run(var.assign(data))
            print(layer_name + ' done.')
        print('Load all trained parameters done!')

    def train_model(self, session, training_data, learning_rate, max_epoach, writer,
        save_step = 100, saver = None, keep_prob = 0.5, save_model_path = ''):
        step = 0
        while True:
          if training_data.train.epochs_completed >= max_epoach:
            break
          step += 1
          print('Iteration: {}, Epoch: {}'.format(step, training_data.train.epochs_completed))

          cur_im, cur_label, cur_mask = training_data.train.next_image()
          _, summary_all = session.run([self.train_step, self.merged_summary], feed_dict={self.X: cur_im, self.Y_:cur_label, self.MASK_:cur_mask, 
            self.KEEP_PROB: keep_prob, self.LEARNING_RATE: learning_rate})

          writer.add_summary(summary_all, step)

          if step % save_step == 0:
            # s = sess.run(merged_summary, feed_dict={x:cur_im, y_: cur_label, mask_:cur_mask, keep_prob: 1.0})
            # writer.add_summary(s, training_data.train.epochs_completed*100 + step)
            train_accuracy = session.run(self.accuracy, feed_dict={self.X: cur_im, self.Y_:cur_label, 
                self.MASK_:cur_mask, self.KEEP_PROB: 1.0})

            test_accuracy, test_cnt = 0, 0
            while True:
              if training_data.validate.epochs_completed >= 1:
                training_data.validate.set_epochs_completed(0)
                break
              test_im, test_label, test_mask = training_data.validate.next_image()
              test_cnt += 1
              test_accuracy += session.run(self.accuracy, feed_dict={self.X: test_im, self.Y_:test_label, 
                self.MASK_:test_mask, self.KEEP_PROB: 1.0})
            test_accuracy /= test_cnt
            saver.save(session, save_model_path + 'my-model', global_step = step)
            print('step {}, training accuracy {}, testing accuracy {}'.format(step, train_accuracy, test_accuracy)) 

    def test_model(self, session, test_data, save_result_path):
        result = tf.nn.softmax(self.dconv3)
        print('Testing ...')
        cnt = 0
        while True:
          cur_image = test_data.next_image()
          if cur_image is None:
            break
          print(cur_image.shape)
          test_result = session.run(result, feed_dict={self.X: cur_image, self.KEEP_PROB: 1.0})
          scipy.io.savemat(save_result_path + 'test_FCN_result_' + "%03d" % (cnt + 1) + '.mat', mdict = {'resultList': np.squeeze(test_result)})
          cnt += 1 
        print('Finish.')

        
def conv(x, filter_height, filter_width, num_filters, name, stride_x = 1, stride_y = 1, padding = 'SAME', relu = True):
    input_channel = int(x.shape[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channel, num_filters])
        # biases = tf.get_variable('biases', shape = [num_filters])
        weights = tf.Variable(tf.random_normal([filter_height, filter_width, input_channel, num_filters], stddev=0.02), name='weights')
        biases = tf.Variable(tf.random_normal([num_filters], stddev=0.02), name='biases')

        conv = convolve(x, weights)
        bias = tf.nn.bias_add(conv, biases)
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        
        if relu:
            relu = tf.nn.relu(bias, name = scope.name)
            return relu
        else:
            return bias

def dconv(x, fuse_x, filter_height, filter_width, name, output_shape = [], output_channels = 1, stride_x = 2, stride_y = 2, padding = 'SAME'):
    input_channels = int(x.shape[-1])
    
    if fuse_x is not None:
        output_shape = tf.shape(fuse_x)
        output_channels = int(fuse_x.shape[-1])

    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [filter_height, filter_width, output_channels, input_channels])
        # biases = tf.get_variable('biases', shape = [output_channels])
        weights = tf.Variable(tf.random_normal([filter_height, filter_width, output_channels, input_channels], stddev=0.02), name='weights')
        biases = tf.Variable(tf.random_normal([output_channels], stddev=0.02), name='biases')

        dconv = tf.nn.conv2d_transpose(x, weights, output_shape = output_shape, strides=[1, stride_y, stride_x, 1], padding = padding, name = scope.name)
        bias = tf.nn.bias_add(dconv, biases)
        if fuse_x is not None:
            fuse = tf.add(bias, fuse_x, name = 'fuse')
            return fuse
        else:
            return bias


def fc(x, num_in, num_out, name, relu = True):
    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [num_in, num_out], trainable = True)
        # biases = tf.get_variable('biases', shape = [num_out], trainable = True)
        weights = tf.Variable(tf.random_normal([num_in, num_out], stddev=0.02), name='weights')
        biases = tf.Variable(tf.random_normal([num_out], stddev=0.02), name='biases')

        act = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)

        if relu:
            relu = tf.nn.relu(act)
            return act, relu
        else:
            return act, []

def max_pool(x, name, filter_height = 2, filter_width = 2, stride_x = 2, stride_y = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1], 
                          strides = [1, stride_y, stride_x, 1], 
                          padding = padding, name = name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def apply_mask(input_matrix, mask):
    return tf.dynamic_partition(input_matrix, mask, 2)[1]

def change_parameter_for_FCN(session, load_path, load_name, save_path):
    weights_dict = {}
    saver = tf.train.Saver()
    saver.restore(session, load_path + load_name)
    weights_dict['conv1'] = [session.run('conv1/weights:0'), session.run('conv1/biases:0')]
    weights_dict['conv2'] = [session.run('conv2/weights:0'), session.run('conv2/biases:0')]
    weights_dict['conv3'] = [session.run('conv3/weights:0'), session.run('conv3/biases:0')]
    weights_dict['conv4'] = [session.run('conv4/weights:0'), session.run('conv4/biases:0')]
    weights_dict['fc1'] = [session.run((tf.reshape(session.run('fc1/weights:0'), [2,2,128,128]))), session.run('fc1/biases:0')]
    weights_dict['fc2'] = [session.run(tf.reshape(session.run('fc2/weights:0'), [1,1,128,2])), session.run('fc2/biases:0')]
    np.save(save_path + 'DeepContourDeepFeature_FCN', weights_dict)

def main(_):
    num_channels = 1
    num_classes = 2

    load_path = save_path = 'D:\\Qian\\TestData\\Test_FCN\\Trained\\'

    x = tf.placeholder(tf.float32, [None, None, None, num_channels])
    keep_prob = tf.placeholder(tf.float32)
    model = DeepContour(x, keep_prob, num_classes, [], False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        change_parameter_for_FCN(sess, load_path, 'my-model-5635', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)