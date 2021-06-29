from tensorflow.keras.models import Model
from keras.applications.xception import preprocess_input, decode_predictions
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend
from src.gradcam import GradCAM


class XceptionWrapper_public():
  def __init__(self, include_top, weights, labels_path, gradcam_layer=None):
      self.model = tf.keras.applications.Xception(include_top=include_top, weights=weights)
      if not gradcam_layer is None:
        self.gradcam_layer = gradcam_layer
      else:
          self.gradcam_layer = self.find_target_layer()
      self.layers = self.model.layers
      self.sess_array = backend.get_session()
      self.w = self.sess_array.run(self.model.weights[234])
      self.b = self.sess_array.run(self.model.weights[235])
      self.find_target_layer_idx()

      GradcamModel = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(self.gradcam_layer).output, self.model.output]
      )
      self.gradCAM_model = GradCAM(self.model, self.gradcam_layer, GradcamModel, self.sess_array)


      with open(labels_path, 'r') as f:
          self.labels = json.load(f)



  def get_image_shape(self):
      return (299,299)


  def run_examples(self, images, BOTTLENECK_LAYER):
      new_model = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(BOTTLENECK_LAYER).output, self.model.output]
      )
      x = (images * 255).copy()
      x = tf.cast(preprocess_input(x), tf.float32)
      (LayerOuts, preds) = new_model(x)
      return self.sess_array.run(LayerOuts)


  def label_to_id(self, CLASS_NAME):
      return int(self.labels[CLASS_NAME.replace(' ', '_')])


  def get_gradient(self, activations, CLASS_ID, BOTTLENECK_LAYER, x):
      gradModel = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(BOTTLENECK_LAYER).output, self.model.output]
      )
      inputs = tf.cast(np.expand_dims(x, axis=0), tf.float32)
      (convOuts, preds) = gradModel(inputs)  # preds after softmax
      loss = preds[:, CLASS_ID[0]]
      grads = tf.gradients(loss, convOuts)
      return -1*self.sess_array.run(grads)[0]


  def find_target_layer(self):
      for layer in reversed(self.model.layers):
          if 'conv' in layer.name:
              return layer.name
      raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")


  def find_target_layer_idx(self):
      self.target_layer_idx = {}
      for idx, layer in enumerate(self.model.layers):
          self.target_layer_idx[layer.name] = idx


  def get_linears(self, x):
      pool_value = self.run_examples(x, 'avg_pool')
      res = pool_value.dot(self.w) + self.b
      return res