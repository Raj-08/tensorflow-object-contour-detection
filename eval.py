import numpy as np
import tensorflow as tf
from model_contour import build_model
slim = tf.contrib.slim
flags = tf.app.flags
from tensorflow.python.ops import variables
from utils import random_crop_and_pad_image
import os
import cv2
import time

FLAGS = flags.FLAGS
flags.DEFINE_integer('eval_crop_size', 480,
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_string('checkpoint', None,
                    'The initial checkpoint in tensorflow format.')


flags.DEFINE_string('image_dir', None,
                    'The Image Directory.')

flags.DEFINE_string('save_preds',None,
                    'Path to folder where predictions will be saved.')

flags.DEFINE_string('eval_text', None,
                    'The Path to the text file containing names of Images and Labels')###This text file should not have extensions in their names such as 8192.png or 8192.jpg instead just the name such as 8192



Image_directory = '/home/ubuntu/gfav/deeplab/tensorflow_deeplab_resnet/data/pascal/VOCdevkit/VOC2007/JPEGImages/'
my_log_dir='./logs'
    
def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
    
image_ph = tf.placeholder(tf.uint8,[1,None,None,3],name='image_placeholder')
size = FLAGS.train_crop_size
image,label=random_crop_and_pad_image(tf.squeeze(image_ph),size,size)
norm_image = tf.image.per_image_standardization(tf.squeeze(image))
norm_image = tf.expand_dims(norm_image,dim=0)
pred = build_model(norm_image)
restore_var =  tf.trainable_variables()
pred = tf.nn.sigmoid(pred)
loader = tf.train.Saver(var_list=restore_var)

init = variables.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    load(loader, sess, FLAGS.checkpoint)
    f = open(FLAGS.eval_text,'r')
    message = f.read()
    lines = message.split('\n')
    for l in lines:
        try :    
            input_image = cv2.imread(Image_directory+l+'.jpg')
            feed_dict={image_ph:input_image}
            P= sess.run(pred, feed_dict=feed_dict)
            np.save(save_preds+l,P)
        except:
            print("ERROR")