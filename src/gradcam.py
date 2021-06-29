from src.ace_helpers import *
from tensorflow.keras.models import Model
import sys
import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions



TARGET_SIZE = (299,299)


class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, gradcam_layer, GradcamModel, sess_array):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = gradcam_layer
        self.GradcamModel = GradcamModel
        self.sess_array = sess_array


    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name:
                return layer.name
        raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")


    def compute_heatmap(self, x, classIdx, upsample_size, keep_percent=50, eps=1e-5):
        # record operations for automatic differentiation
        inputs = tf.cast(x, tf.float32)
        (convOuts, preds) = self.GradcamModel(inputs)  # preds after softmax
        loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tf.gradients(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        cam = cam.eval()

        # convert to 3D
        # Apply reLU
        cam3 = np.maximum(cam, 0)
        cam3 = cam3 / np.max(cam3)
        cam3 = cv2.resize(cam3, upsample_size, cv2.INTER_LINEAR)
        cam3 = np.expand_dims(cam3, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        # cam = np.where(cam3 > np.median(cam3), 1, 0)
        cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
        return cam3, cam


    def overlay_gradCAM(self, img, cam3, cam):
        new_img = cam * img
        new_img = new_img.astype("uint8")

        cam3 = np.uint8(255 * cam3)
        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
        new_img_concat = 0.3 * cam3 + 0.5 * img
        new_img_concat = (new_img_concat * 255.0 / new_img_concat.max()).astype("uint8")

        return new_img, new_img_concat


    def showCAMs(self, img, x, chosen_class, upsample_size):
        plt.imshow(img.astype("uint8"))
        plt.show()

        cam3, cam = self.compute_heatmap(x=x, classIdx=chosen_class, upsample_size=upsample_size)
        new_img, new_img_concat = self.overlay_gradCAM(img, cam3, cam)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img)
        plt.show()

        new_img_concat = cv2.cvtColor(new_img_concat, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img_concat)
        plt.show()


def get_preds(model, IMAGE_PATH):
    img = image.load_img(IMAGE_PATH, target_size=TARGET_SIZE)
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0).copy()
    x = preprocess_input(x)
    preds = model.predict(x)
    return img, x, preds