import tensorflow as tf
from ops import conv,batch_normalization,unpool_with_argmax,sigmoid,relu
is_training = True
def build_model(input_img):
    
    conv1=conv(input_img,7, 7, 64, 1, 1, biased=False, relu=False, name='conv1')
    bn_conv1=batch_normalization(conv1,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')    
    conv2=conv(bn_conv1,3, 3, 64, 1, 1, biased=False, relu=False, name='conv2')
    bn_conv2=batch_normalization(conv2,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2')
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(bn_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print(pool1)
    print('INd',pool1_indices)
    
    conv3=conv(pool1,3, 3,128, 1, 1, biased=False, relu=False, name='conv3')
    bn_conv3=batch_normalization(conv3,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3')
    conv4=conv(bn_conv3,3, 3, 128, 1, 1, biased=False, relu=False, name='conv4')
    bn_conv4=batch_normalization(conv4,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4')
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(bn_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print(pool2)
    print('INd',pool2_indices)
    conv5=conv(pool2,3, 3, 256, 1, 1, biased=False, relu=False, name='conv5')
    bn_conv5=batch_normalization(conv5,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5')    
    conv6=conv(bn_conv5,3, 3, 256, 1, 1, biased=False, relu=False, name='conv6')
    bn_conv6=batch_normalization(conv6,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6')
    conv7=conv(bn_conv6,3, 3, 256, 1, 1, biased=False, relu=False, name='conv7')
    bn_conv7=batch_normalization(conv7,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7')
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(bn_conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    conv8=conv(pool3,3, 3, 512, 1, 1, biased=False, relu=False, name='conv8')
    bn_conv8=batch_normalization(conv8,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8')    
    conv9=conv(bn_conv8,3, 3, 512, 1, 1, biased=False, relu=False, name='conv9')
    bn_conv9=batch_normalization(conv9,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9')
    conv10=conv(bn_conv9,3, 3, 512, 1, 1, biased=False, relu=False, name='conv10')
    bn_conv10=batch_normalization(conv10,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10')
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(bn_conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    conv11=conv(pool4,3, 3, 512, 1, 1, biased=False, relu=False, name='conv11')
    bn_conv11=batch_normalization(conv11,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11')    
    conv12=conv(bn_conv11,3, 3, 512, 1, 1, biased=False, relu=False, name='conv12')
    bn_conv12=batch_normalization(conv12,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12')
    conv13=conv(bn_conv12,3, 3, 512, 1, 1, biased=False, relu=False, name='conv13')
    bn_conv13=batch_normalization(conv13,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv13')
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(bn_conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    conv14=conv(pool5,3, 3, 4096, 1, 1, biased=False, relu=False, name='conv14')
    bn_conv14=batch_normalization(conv14,is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv14')       
    deconv6=conv(bn_conv14,1, 1, 512, 1, 1, biased=False, relu=False, name='conv15')
    bn_deconv6=batch_normalization(deconv6,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv6')    
    #print(bn_deconv6)
    unpool_5 = unpool_with_argmax(bn_deconv6, ind=pool5_indices, name="unpool_5")
    deconv5=conv(unpool_5,5, 5, 512, 1, 1, biased=False, relu=False, name='deconv5')
    bn_deconv5=batch_normalization(deconv5,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv5')
    
    #print(bn_deconv5)
    unpool_4 = unpool_with_argmax(bn_deconv5, ind=pool4_indices, name="unpool_4")
    deconv4=conv(unpool_4,5, 5, 256, 1, 1, biased=False, relu=False, name='deconv4')
    bn_deconv4=batch_normalization(deconv4,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv4')
    unpool_3 = unpool_with_argmax(bn_deconv4, ind=pool3_indices, name="unpool_3")
    deconv3=conv(unpool_3,5, 5,128, 1, 1, biased=False, relu=False, name='deconv3')
    bn_deconv3=batch_normalization(deconv3,is_training=is_training, activation_fn=tf.nn.relu, name='bn_decon3')
    print(bn_deconv3)
    unpool_2 = unpool_with_argmax(bn_deconv3, ind=pool2_indices, name="unpool_2")
    deconv2=conv(unpool_2,5, 5,64, 1, 1, biased=False, relu=False, name='deconv2')
    bn_deconv2=batch_normalization(deconv2,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv2')
    print(bn_deconv2)
    unpool_1 = unpool_with_argmax(bn_deconv2, ind=pool1_indices, name="unpool_1")
    deconv1=conv(unpool_1,5, 5,32, 1, 1, biased=False, relu=False, name='deconv1')
    bn_deconv1=batch_normalization(deconv1,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv1')
    deconv0=conv(bn_deconv1,5, 5, 1, 1, 1, biased=False, relu=False, name='deconv0')
   # bn_deconv0=batch_normalization(deconv0,is_training=is_training, activation_fn=tf.nn.relu, name='bn_deconv0')
   # pred = sigmoid(bn_deconv0,name='pred')
    
    return deconv0;








