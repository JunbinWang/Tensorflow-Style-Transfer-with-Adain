import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
import utils
import os

from encoder import Encoder
from decoder import Decoder
from adain_norm import AdaIN

VGG_PATH = 'vgg19_normalised.npz'
DECODER_PATH = 'decoder.npy'


content_path = 'images/content/'
style_path ='images/style/'
output_path = './output/'


if __name__ == '__main__':


    content_images = os.listdir(content_path)
    style_images = os.listdir(style_path)

    with tf.Graph().as_default(), tf.Session() as sess:

        # 初始化对象
        encoder = Encoder(VGG_PATH)
        decoder = Decoder(mode='test', weights_path='decoder.npy')

        content_input = tf.placeholder(tf.float32, shape=(1,None,None,3), name='content_input')
        style_input =   tf.placeholder(tf.float32, shape=(1,None,None,3), name='style_input')

        # switch RGB to BGR
        content = tf.reverse(content_input, axis=[-1])
        style   = tf.reverse(style_input, axis=[-1])

        # preprocess image
        content = encoder.preprocess(content)
        style   = encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = encoder.encode(content)
        enc_s, enc_s_layers = encoder.encode(style)

        # pass the encoded images to AdaIN
        target_features = AdaIN(enc_c, enc_s)

        # decode target features back to image
        generated_img = decoder.decode(target_features)

        # deprocess image
        generated_img = encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        sess.run(tf.global_variables_initializer())

        for s in style_images:
            for c in content_images:
                # Load image from path and add one extra diamension to it.
                content_image = imread(os.path.join(content_path,c), mode='RGB')
                style_image   = imread(os.path.join(style_path,s), mode='RGB')
                content_tensor = np.expand_dims(content_image, axis=0)
                style_tensor = np.expand_dims(style_image, axis=0)

                result = sess.run(generated_img, feed_dict={content_input: content_tensor,style_input: style_tensor})
                result_name = os.path.join(output_path,s.split('.')[0]+'_'+c.split('.')[0]+'.jpg')
                print(result_name,' is generated')
                imsave(result_name, result[0])


