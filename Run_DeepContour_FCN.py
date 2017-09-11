import numpy as np 
import argparse
import sys, os
import distutils.core

import tensorflow as tf 

from DeepContour import DeepContour
import data_image 

def main(_):
    num_channels = FLAGS.num_channels
    num_classes = FLAGS.num_classes
    is_training = FLAGS.is_training
    max_train_epoach = 50
    learning_rate = 0.0001

    model = DeepContour(num_classes, num_channels, [], is_training, is_FCN = True, weights_path = 'D:\\Qian\\TestData\\Test_FCN\\Trained\\')

    if is_training:
      training_data = data_image.prepare_data_set(FLAGS.train_dir, 0.1, num_channels = num_channels)
      print(training_data.train.sample_mean)
      
      writer = tf.summary.FileWriter(FLAGS.save_model_path)
    else:
      sample_mean = tf.Variable(np.empty(shape=[num_channels], dtype = float), name = 'sample_mean')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if is_training:
          writer.add_graph(sess.graph)
          model.train_model(sess, training_data, learning_rate, max_train_epoach, writer, 
            save_step = 10, saver = saver, save_model_path = FLAGS.save_model_path)
        else:
          model.restore_trained_parameters(sess, saver, FLAGS.save_model_path, 'my-model-3400')
          test_data = data_image.TestImage(FLAGS.test_directory, 0, sample_mean = sess.run(sample_mean), num_channels = num_channels)
          model.test_model(sess, test_data, FLAGS.save_result_path)

        if is_training:
          writer.close()
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_model_path',
      type=str,
      default= 'D:\\Qian\\TestData\\Test_FCN\\Trained_FCN\\subtract_mean\\',
      help='Directory for storing training data'
  )
  parser.add_argument(
      '--test_directory',
      type=str,
      default = 'D:\\GoogleDrive_Qian\\Foram\\testing\\original\\',
      help='Directory for storing test data'
  )
  parser.add_argument(
      '--save_result_path',
      type=str,
      default = 'D:\\Qian\\TestData\\Test_FCN\\FCN_result\\subtract_mean\\3\\',
      help='Directory for storing result'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\',
      help='Directory for storing training data'
  )
  parser.add_argument(
      '--is_training', 
      type = lambda x:bool(distutils.util.strtobool(x)),
      default = True,
      help = 'Training or testing'
  )
  parser.add_argument(
      '--num_channels',
      type=int,
      default=1,
      help='Number of input channel'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=2,
      help='Number of classes'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)