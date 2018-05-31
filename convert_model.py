import tensorflow as tf
import numpy as np

#  Convert a trained model in tensorflow format to npy file
#
#  Decoder variable from checkpoint file (in order)
#
# conv4_1/kernel
# conv4_1/bias
# conv3_4/kernel
# conv3_4/bias
# conv3_3/kernel
# conv3_3/bias:0
# conv3_2/kernel
# conv3_2/bias
# conv3_1/kernel
# conv3_1/bias
# conv2_2/kernel
# conv2_2/bias
# conv2_1/kernel
# conv2_1/bias:0
# conv1_2/kernel
# conv1_2/bias
# conv1_1/kernel
# conv1_1/bias

if __name__ == '__main__':


    weight_vars = []
    with tf.Session() as sess:
      decoder_vars = dict()
      saver = tf.train.import_meta_graph('./models/-60000.meta')
      saver.restore(sess, tf.train.latest_checkpoint('./models/'))
      tvs = [v for v in tf.trainable_variables()]
      for v in tvs:
        print(v.name[8:-2])
        decoder_vars[v.name[8:-2]] = sess.run(v)

      np.save('decoder.npy', decoder_vars)

      print('decoder variable is saved to decoder.npy now \n')

      # print('reload it from file now')

      # decoder = np.load('decoder.npy')[()]

      # print(decoder)