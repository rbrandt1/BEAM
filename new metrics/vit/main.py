import sys
root_url = "./"

import sys
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers
from google.colab.patches import cv2_imshow

from dataset_synth_newGT import dataset_synth_newGT
from xai_metrics.Our_CompactnessMetric import Our_CompactnessMetric
from xai_metrics.Our_CompletenessMetric import Our_CompletenessMetric
from xai_metrics.Our_CorrectnessMetric import Our_CorrectnessMetric

class IdentityAttention(keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(IdentityAttention, self).__init__(**kwargs)
        self.embed_dim = 144*3
        self.position_dim = 9

    def build(self):
        self.wq = self.add_weight(shape=(self.embed_dim, self.embed_dim),
                                    initializer='identity',
                                    trainable=True)
        self.wk = self.add_weight(shape=(self.embed_dim, self.embed_dim),
                                    initializer='identity',
                                    trainable=True)
        self.wv = self.add_weight(shape=(self.embed_dim, self.embed_dim),
                                    initializer='identity',
                                    trainable=True)
        self.wo = self.add_weight(shape=(self.embed_dim, self.embed_dim),
                                    initializer='identity',
                                    trainable=True)

    def get_config(self):
        config = super().get_config()
        return config

    def get_mask(self):
        i = tf.range(self.position_dim)[:, tf.newaxis]
        j = tf.range(self.position_dim)
        mask = tf.cast(tf.equal(i, j), dtype=tf.float32)  # 1 on diagonal, 0 elsewhere
        mask = (1.0 - mask) * -1e9
        mask = tf.reshape(mask, (self.position_dim, self.position_dim))
        return mask


    def call(self, inputs):
        query = tf.matmul(inputs, self.wq)
        key = tf.matmul(inputs, self.wk)
        value = tf.matmul(inputs, self.wv)

        scores = tf.matmul(query, key, transpose_b=True)

        mask = self.get_mask()
        scores = scores + mask

        attention_weights = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(attention_weights, value)

        context = tf.matmul(context, self.wo)

        return context

class IdentityNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(IdentityNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IdentityNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[1,2], keepdims=True)

        # Normalize
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + 1e-9)

        # Apply scale (gamma) and offset (beta)
        normalized_inputs *= tf.sqrt(variance + 1e-9)   # gamma = tf.sqrt(variance + 1e-9)
        normalized_inputs += mean                       # beta = mean

        return normalized_inputs

    def get_config(self):
        config = super(IdentityNormalization, self).get_config()
        return config




class TransformerEncoder_cp(layers.Layer):
    def __init__(self, **kwargs):
        super(TransformerEncoder_cp, self).__init__(**kwargs)
        embed_dim = 144*3
        dropout=0.2

        self.ffn_layers_encoderblock_cp = [tf.keras.layers.Dense(144*3),tf.keras.layers.ReLU(max_value=1)]
        self.ffn_encoderblock_cp = keras.Sequential(self.ffn_layers_encoderblock_cp)
        self.ffn = self.ffn_encoderblock_cp

        self.attention_layer = IdentityAttention()
        self.normalization_layer1 = IdentityNormalization()
        self.normalization_layer2 = IdentityNormalization()
        self.dropoutlayer1 = layers.Dropout(dropout)
        self.dropoutlayer2 = layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()

        return config

    def call(self, inputs):
        attn_output = self.attention_layer(inputs)
        attn_output = self.dropoutlayer1(attn_output, training=False)

        out1 = self.normalization_layer1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropoutlayer2(ffn_output, training=False)

        out1 = self.normalization_layer2(out1 + ffn_output)

        return out1


class TransformerEncoder_concepts(layers.Layer):
    def __init__(self, **kwargs):
        super(TransformerEncoder_concepts, self).__init__(**kwargs)
        embed_dim = 144*3
        dropout=0.2

        self.ffn_layers_encoderblock_concepts = [tf.keras.layers.Dense(8,  name='conv1', use_bias=True),tf.keras.layers.ReLU(max_value=1),
                                            tf.keras.layers.Dense(8*2, name='conv2', use_bias=True),tf.keras.layers.ReLU(max_value=1),
                                            tf.keras.layers.Dense(144*3, name='conv3', use_bias=True),tf.keras.layers.ReLU(max_value=1)]
        self.ffn_encoderblock_concepts = keras.Sequential(self.ffn_layers_encoderblock_concepts)
        self.ffn = self.ffn_encoderblock_concepts

        self.attention_layer = IdentityAttention()
        self.normalization_layer1 = IdentityNormalization()
        self.normalization_layer2 = IdentityNormalization()
        self.dropoutlayer1 = layers.Dropout(dropout)
        self.dropoutlayer2 = layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        attn_output = self.attention_layer(inputs)
        attn_output = self.dropoutlayer1(attn_output, training=False)

        out1 = self.normalization_layer1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropoutlayer2(ffn_output, training=False)

        out1 = self.normalization_layer2(out1 + ffn_output)

        return out1


 # **** Model ***

def genModel(conceptid):

  input_tensor = keras.Input(shape=(18, 18, 3))
  x = input_tensor

  # embedding (Conv2D)

  embedding_layer = layers.Conv2D(filters=144*3,kernel_size=6, padding="valid", strides=6,use_bias=False, name="embedding")
  x = embedding_layer(x)

  # reshape (Reshape)

  x = layers.Reshape(target_shape=(9,144*3),name="reshape")(x)


  # encoderblock_cp  (TransformerBlock)

  encoderblock_cp = TransformerEncoder_cp()
  x = encoderblock_cp(x)


  # encoderblock_concepts  (TransformerBlock)

  encoderblock_concepts = TransformerEncoder_concepts()
  x = encoderblock_concepts(x)


  # Dense

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(30, name='dense0', use_bias=True)(x)
  x = tf.keras.layers.ReLU(max_value=1)(x)
  x = tf.keras.layers.Dense(60, name='dense1', use_bias=True)(x)
  x = tf.keras.layers.ReLU(max_value=1)(x)
  x = tf.keras.layers.Dense(5, name='dense2', use_bias=True)(x)
  x = tf.keras.layers.ReLU(max_value=1)(x)


  model = keras.Model(inputs=input_tensor, outputs=x)
  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

  model.summary()




  # *** Set Weights ***

  offset2 = 144*2
  offset = 144

  # embedding_layer

  weights = embedding_layer.get_weights()
  weights[0] = np.zeros_like(weights[0])

  for y in range(6):
    for x in range(6):
      for c in range(3):
        out_idx = (y * 6 + x) * 3 + c
        weights[0][y, x, c, out_idx+offset2] = 1

  embedding_layer.set_weights(weights)


  ###### encoderblock_cp

  # fully connected


  weights = encoderblock_cp.ffn_layers_encoderblock_cp[0].get_weights()
  weights[0] = np.zeros_like(weights[0])
  weights[1] = -1 * np.ones_like(weights[1])

  for y in range(6):
    for x in range(6):
      for c in range(3):
        for pc in range(12):

          in_idx = (y * 6 + x) * 3 + c
          out_idx = int(((int(y/3) * 6/3 + int(x/3)) * 3 + c )*12 + pc)

          weights[0][in_idx+offset2, out_idx+offset] = dataset_synth_newGT.gen_concept_part_weights(pc)[y % 3][x % 3] * 0.5

  encoderblock_cp.ffn_layers_encoderblock_cp[0].set_weights(weights)


  ###### encoderblock_concepts



  # ****              CONV1: a&b                   ****

  candidates = [["and",7,0,0,   10,0,1],
                ["and",4,1,0,   1,1,1],

                ["and",6,0,0,   9,0,1],
                ["and",3,1,0,   0,1,1],

                ["or",7+12*1,0,0,   10+12*1,0,1],

                ["and",8+12*1,0,0,   11+12*1,0,1],

                ["or",8+12*1,0,0,   11+12*1,0,1],

                ["and",4+12*2,1,0,   1+12*2,1,1]]

  weigths = encoderblock_concepts.ffn_layers_encoderblock_concepts[0].get_weights()
  weigths[0] = np.zeros_like(weigths[0])
  weigths[1] = np.zeros_like(weigths[1])

  for operator in ["and","or"]:
      for left in range(12*3):
          for right in range(12*3):
              for y_left in range(2):
                  for x_left in range(2):
                      for y_right in range(2):
                          for x_right in range(2):

                              if [operator,  left, y_left, x_left,      right, y_right, x_right ] in candidates:

                                  index = [str(x) for x in candidates].index(str([operator,  left, y_left, x_left,      right, y_right, x_right ]))

                                  if operator == "and":
                                      newindex = (y_left*2 + x_left)*3*12 + left # y_left, x_left, left
                                      weigths[0][newindex+offset, index] = 1.0 * 0.5
                                      newindex = (y_right*2 + x_right)*3*12 + right # y_right, x_right, right
                                      weigths[0][newindex+offset, index] = 1.0 * 0.5

                                      weigths[1][index] = -1.0

                                  elif operator == "or":
                                      newindex = (y_left*2 + x_left)*3*12 + left # y_left, x_left, left
                                      weigths[0][newindex+offset, index] = 1.0 * 0.5
                                      newindex = (y_right*2 + x_right)*3*12 + right # y_right, x_right, right
                                      weigths[0][newindex+offset, index] = 1.0 * 0.5

                                      weigths[1][index] = 0.0

  encoderblock_concepts.ffn_layers_encoderblock_concepts[0].set_weights(weigths)




  # ****              CONV2: c&d                  ****

  weigths = encoderblock_concepts.ffn_layers_encoderblock_concepts[2].get_weights()
  weigths[0] = np.zeros_like(weigths[0])
  weigths[1] = np.zeros_like(weigths[1])

  for id in range(8):
      weigths[0][id, id] = 1.0
      weigths[1][id] = 0

      weigths[0][id, id+8] = -1.0
      weigths[1][id+8] = 1.0

  encoderblock_concepts.ffn_layers_encoderblock_concepts[2].set_weights(weigths)




  # ****              CONV3: e                  ****


  weigths = encoderblock_concepts.ffn_layers_encoderblock_concepts[4].get_weights()
  weigths[0] = np.zeros_like(weigths[0])
  weigths[1] = np.zeros_like(weigths[1])


  weigths[0][0, 0] = 1
  weigths[0][1, 0] = 1
  weigths[1][0] = 0

  weigths[0][2, 1] = 1
  weigths[0][3, 1] = 1
  weigths[1][1] = -1

  weigths[0][4, 2] = 1
  weigths[0][7, 2] = 1
  weigths[1][2] = -1

  weigths[0][5, 3] = 1

  weigths[0][6, 4] = 1
  weigths[0][5+8, 4] = 1
  weigths[1][4] = -1


  encoderblock_concepts.ffn_layers_encoderblock_concepts[4].set_weights(weigths)



  #  Dense0:  (None, 45) (concept vector top left, '' top middle, '' top right, etc) -> (45,30)

  weigths = model.get_layer('dense0').get_weights()
  weigths[0][:,:] = 0 # (in node, out node)
  weigths[1][:] = 0

  idx = 0

  for concept_id in range(5):

    weigths[0][concept_id+0*144*3, idx] = 1 # pos 0
    weigths[1][idx] = 0
    idx += 1


    weigths[0][concept_id+3*144*3, idx] = 1 # pos 3
    idx += 1

    weigths[0][concept_id+1*144*3, idx] = 1 # pos 1 AND pos 2
    weigths[0][concept_id+2*144*3, idx] = 1
    weigths[1][idx] = -1
    idx += 1


    weigths[0][concept_id+1*144*3, idx] = 1 # pos 1 OR pos 2
    weigths[0][concept_id+2*144*3, idx] = 1
    weigths[1][idx] = 0
    idx += 1


    weigths[0][concept_id+0*144*3, idx] = 1 # pos 0 OR pos 1
    weigths[0][concept_id+1*144*3, idx] = 1
    weigths[1][idx] = 0
    idx += 1


    weigths[0][concept_id+2*144*3, idx] = 1# pos 2 OR pos 3
    weigths[0][concept_id+3*144*3, idx] = 1
    weigths[1][idx] = 0
    idx += 1


  model.get_layer('dense0').set_weights(weigths)


  # ****              Dense1: c&d                  ****


  weigths = model.get_layer('dense1').get_weights()
  weigths[0][:,:] = 0 # (in node, out node)
  weigths[1][:] = 0


  for id in range(30):
      weigths[0][id, id] = 1.0
      weigths[1][id] = 0.0

      weigths[0][id, id+30] = -1.0
      weigths[1][id+30] = 1.0

  model.get_layer('dense1').set_weights(weigths)


  # ****              Dense2: e                  ****

  weigths = model.get_layer('dense2').get_weights()
  weigths[0][:,:] = 0 # (in node, out node)
  weigths[1][:] = 0

  weigths[0][0+conceptid*6, 0] = 1
  weigths[1][0] = 0

  weigths[0][1+conceptid*6, 1] = 1
  weigths[1][1] = 0

  weigths[0][2+conceptid*6, 2] = 1
  weigths[1][2] = 0

  weigths[0][3+conceptid*6, 3] = 1
  weigths[0][2+30+conceptid*6, 3] = 1
  weigths[1][3] = -1

  weigths[0][4+30+conceptid*6, 4] = 1
  weigths[0][5+30+conceptid*6, 4] = 1
  weigths[1][4] = -1

  model.get_layer('dense2').set_weights(weigths)



  model.compile(optimizer='adam', loss='mse')


  return model

# **** Examples ***
for conceptid in range(5):

  print("conceptid",conceptid)

  inputdata = dataset_synth_newGT.generate(4, conceptid)

  model = genModel(conceptid)

  output = model(inputdata['test']['x'])
  cv2_imshow(cv2.resize(inputdata['test']['x'][0]*255, (64, 64),interpolation=cv2.INTER_NEAREST) )

  output = np.array(output)
  output[output < 0.000001] = 0
  output[output > 1-0.000001] = 1
  [print(np.array(x)) for x in output]

import pandas as pd
from skimage.metrics import structural_similarity
import sklearn.metrics
import cv2
import time

from xplique.attributions import DeconvNet,GradCAM,GradCAMPP,GradientInput,GuidedBackprop,IntegratedGradients,Saliency,SmoothGrad,SquareGrad,VarGrad,KernelShap,Lime,Occlusion,Rise
from xplique.metrics import MuFidelity,Deletion,Insertion


settings = {}

settings['num_examples_per_class'] = 16
settings['normalize_explanations'] = True

settings['saveSelectedImages'] = True
settings['saveAllImages'] = False
settings['resize_Images'] = False

# ---------------------- Functions ---------------------- #

def checkNum(num):
    if (num == None) or (not np.isreal(num)) or (np.isnan(num)):
        print("ERROR checkNum: ",num)
        exit()

def checkList(list_var):

    for num in list_var:
        checkNum(num)

def saveImages(path,images,conceptid, applyColormap,data,postfix=""):

    for img,id in zip(images,range(len(images))):

        if settings['resize_Images']:
            img2 = cv2.resize(img,(72,72), interpolation = cv2.INTER_NEAREST)
        else:
            img2 = img

        if applyColormap:
            saveImageColomap(path+str(conceptid)+"_"+str(np.argmax(data['test']['y'][id]))+"_"+str(id)+postfix+".png", img2)
        else:
            saveImage(path+str(conceptid)+"_"+str(np.argmax(data['test']['y'][id]))+"_"+str(id)+postfix+".png", img2)

def saveImage(path, image):
    cv2.imwrite(path, image*255)


def saveImageColomap(path, image):
    if image.shape[-1] == 3:
        image2 = dataset_synth_newGT.convert_to_2D(image)
    else:
        image2 = image

    scaled = (((image2 + 1)/2.0)*255).astype(np.uint8)
    im_color = cv2.applyColorMap(scaled, cv2.COLORMAP_CIVIDIS)
    cv2.imwrite(path, im_color)


def saveColormap():

    img = np.zeros((256,10))

    for i in range(0,256):
        img[i,:] = (i/255.0)*2-1

    saveImageColomap(root_url+"paper_images/selected/colormap.png", img)

def startTimer():
    return time.time()

def endTimer(start_time, num_items):
    return ((time.time() - start_time)*1000.0) / num_items


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def flatten_and_turn_binary(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])), dtype='bool')
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])), dtype='bool')

    for idx_iter in range(gt_explanation_copy.shape[0]):

        gt_binary = np.zeros(gt_explanation_copy[idx_iter].shape, dtype='bool')
        gt_binary[gt_explanation_copy[idx_iter] != 0] = True

        gt_explanation_copy_flat[idx_iter] = gt_binary.flatten()


        explanation_binary = np.zeros(explanations_copy[idx_iter].shape, dtype='bool')
        explanation_binary[explanations_copy[idx_iter] != 0] = True

        explanations_copy_flat[idx_iter] = explanation_binary.flatten()

    return explanations_copy_flat, gt_explanation_copy_flat

def turn_binary(explanations_copy, invert):

    if invert:
        explanations_copy_binary = np.ones(explanations_copy.shape, dtype='bool')

        explanations_copy_binary[explanations_copy != 0] = False
    else:
        explanations_copy_binary = np.zeros(explanations_copy.shape, dtype='bool')

        explanations_copy_binary[explanations_copy != 0] = True

    return explanations_copy_binary


def flatten_only(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])))
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])))

    for idx_iter in range(gt_explanation_copy.shape[0]):
        gt_explanation_copy_flat[idx_iter] = gt_explanation_copy[idx_iter].flatten()
        explanations_copy_flat[idx_iter] = explanations_copy[idx_iter].flatten()

    return explanations_copy_flat, gt_explanation_copy_flat


def calc_metrics_for_XAI_method(Xai_method, xai_name, model, data, conceptid):
        print("*** ",xai_name," ***")

        result_data_xaimethod = {}
        result_data_xaimethod_time = {}


        # calc num_items
        if data['test']['x'].shape[0] != data['test']['gt_explanation'].shape[0]:
            print("ERROR! data['test']['x'].shape[0] != data['test']['gt_explanation'].shape[0]")
            exit()
        num_items = data['test']['x'].shape[0]

        # Run xai method and time

        start_time = startTimer()


        explainer = Xai_method(model, batch_size=32)

        explanations = explainer(data['test']['x'], data['test']['y'])
        explanations = explanations.numpy()


        print("determine",explanations.shape)

        xai_method_time = endTimer(start_time, num_items)

        # normalize explanations
        if settings['normalize_explanations']:
            for i in range(len(explanations)):
                explanations[i] = dataset_synth_newGT.normalize_explanation(explanations[i])
        else:
            print("WARNING! NOT NORMALIZING EXPLANATIONS! ")



        # Determine GT: 2D or 3D?



        if explanations.shape[-1] == 3:# 3D GT
            gt_explanation = data['test']['gt_explanation3D']

            print("3D explanations")

            # mufidelity fix
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0][:] = 0.000000001

        else: #2D GT
            gt_explanation = data['test']['gt_explanation']

            print("2D explanations")

            # mufidelity fix
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0] = 0.000000001

        # Calc prior metrics

        for Metric, metric_name in zip(all_xai_metrics,all_xai_metrics_names):
            print(metric_name)

            data_x_copy = data['test']['x'].astype('float32')
            data_y_copy = data['test']['y'].astype('float32')
            explanations_copy = explanations.astype('float32')

            start_time = startTimer()
            explainer = Metric(model, data_x_copy, data_y_copy)
            result_data_xaimethod[metric_name] =  explainer(explanations_copy)
            result_data_xaimethod_time[metric_name] = endTimer(start_time, num_items)

            checkNum(result_data_xaimethod[metric_name])



        # Cosine Similarity (GUIDOTTI2021103428)
        print("CosSim")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')


        explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = np.linalg.norm(np.dot(explanation,gt))/(np.linalg.norm(explanation)*np.linalg.norm(gt))

            scores.append(score)

        result_data_xaimethod_time['CosSim'] = endTimer(start_time, num_items)
        result_data_xaimethod['CosSim'] = sum(scores) / len(scores)

        checkList(scores)




        # Cosine Similarity (GUIDOTTI2021103428)

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')


        explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = np.dot(explanation,gt)/(np.linalg.norm(explanation)*np.linalg.norm(gt))

            scores.append(score)

        result_data_xaimethod_time['_CosSim'] = endTimer(start_time, num_items)
        result_data_xaimethod['_CosSim'] = sum(scores) / len(scores)

        checkList(scores)




        # SSIM
        print("SSIM")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy, explanations_copy):

            if explanation.shape[-1] == 3:
                score = structural_similarity(gt,explanation, channel_axis=-1, data_range=2)
            else:
                score = structural_similarity(gt, explanation, data_range=2)

            scores.append(score)

        result_data_xaimethod_time['SSIM'] = endTimer(start_time, num_items)
        result_data_xaimethod['SSIM'] = sum(scores) / len(scores)

        checkList(scores)




        # CONCISSENESS (Amparore_2021)
        print("CONCISSENESS")

        explanations_copy = explanations.astype('bool')

        start_time = startTimer()
        scores = []
        for explanation in explanations_copy:
            total = np.prod(explanation.shape)

            iszero = (total - np.count_nonzero(explanation))+0.0

            score = iszero / total

            scores.append(score)


        result_data_xaimethod_time['Conciseness'] = endTimer(start_time,num_items )
        result_data_xaimethod['Conciseness'] = sum(scores) / len(scores)

        checkList(scores)



        # F1
        print("F1")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = sklearn.metrics.f1_score(gt, explanation, average='binary', pos_label=1)

            scores.append(score)


        result_data_xaimethod_time['F1'] = endTimer(start_time, num_items)
        result_data_xaimethod['F1'] = sum(scores) / len(scores)

        checkList(scores)





        # MAE
        print("MAE")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = sklearn.metrics.mean_absolute_error(gt, explanation)

            scores.append(score)


        result_data_xaimethod_time['MAE'] = endTimer(start_time, num_items)
        result_data_xaimethod['MAE'] = sum(scores) / len(scores)

        checkList(scores)




        # Intersection over union IoU
        print("IoU")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = sklearn.metrics.jaccard_score(gt, explanation, average='binary', pos_label=1)

            scores.append(score)

        result_data_xaimethod_time['IoU'] = endTimer(start_time, num_items)
        result_data_xaimethod['IoU'] = sum(scores) / len(scores)

        checkList(scores)




        # Precision
        print("Precision")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = sklearn.metrics.precision_score(gt, explanation, average='binary', pos_label=1)

            scores.append(score)


        result_data_xaimethod_time['PR'] = endTimer(start_time, num_items)
        result_data_xaimethod['PR'] = sum(scores) / len(scores)

        checkList(scores)






        # Recall
        print("Recall")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):

            score = sklearn.metrics.recall_score(gt, explanation, average='binary', pos_label=1)

            scores.append(score)


        result_data_xaimethod_time['RE'] = endTimer(start_time, num_items)
        result_data_xaimethod['RE'] = sum(scores) / len(scores)

        checkList(scores)






        # Energy-Based Pointing Game (DBLP:journals/corr/abs-1910-01279)
        print("Energy-Based Pointing Game")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

            top = np.ma.array(abs(explanation), mask = gt).sum()
            if not top:
                top = 0.0

            bottom = abs(explanation).sum()

            if top == 0 and bottom == 0:
                score = 1.0
            elif top == 0:
                score = 0.0
            else:
                score = top / bottom

            scores.append(score)

        result_data_xaimethod_time['EBPG'] = endTimer(start_time, num_items)
        result_data_xaimethod['EBPG'] = sum(scores) / len(scores)

        checkList(scores)




        # Relevance Rank Accuracy (arras2021ground)
        print("Relevance Rank Accuracy")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        gt_explanation_copy_binary = turn_binary(gt_explanation_copy,False)

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

            num_K = np.count_nonzero(gt)

            kth_highest_value = np.partition(abs(explanation).flatten(), -(num_K))[-(num_K)]

            exp_greater_k = np.zeros(explanation.shape, dtype='bool')
            exp_greater_k[abs(explanation) >= kth_highest_value] = True
            exp_greater_k[gt == 0] = False

            score =  exp_greater_k.sum() / num_K

            scores.append(score)

        result_data_xaimethod_time['RRA'] = endTimer(start_time, num_items)
        result_data_xaimethod['RRA'] = sum(scores) / len(scores)

        checkList(scores)
































        for version_soft in ["abs","pos","neg"]:


            # Soft Precision (AttributionLab)
            print("Soft Precision "+version_soft)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0

            gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

                top = np.ma.array(abs(explanation), mask = gt).sum()
                if not top:
                    top = 0.0

                bottom = abs(explanation).sum()

                if top == 0 and bottom == 0:
                    score = 1.0
                elif top == 0:
                    score = 0.0
                else:
                    score = top / bottom

                scores.append(score)

            result_data_xaimethod_time['soft_PR_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['soft_PR_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)




            # Soft Recall (AttributionLab)
            print("Soft Recall "+version_soft)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0


            gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

                top = np.ma.array(abs(explanation), mask = gt).sum()
                if not top:
                    top = 0.0

                bottom = (1-gt).sum()

                if top == 0 and bottom == 0:
                    score = 1.0
                elif top == 0:
                    score = 0.0
                else:
                    score = top / bottom

                scores.append(score)

            result_data_xaimethod_time['soft_RE_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['soft_RE_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)





            # Soft F1 (AttributionLab)
            print("Soft F1 "+version_soft)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0


            gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)

            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):


                # soft precision

                top = np.ma.array(abs(explanation), mask = gt).sum()
                if not top:
                    top = 0.0

                bottom = abs(explanation).sum()

                if top == 0 and bottom == 0:
                    score_softPR = 1.0
                elif top == 0:
                    score_softPR = 0.0
                else:
                    score_softPR = top / bottom



                # soft recall

                bottom = (1-gt).sum()

                if top == 0 and bottom == 0:
                    score_softRE = 1.0
                elif top == 0:
                    score_softRE = 0.0
                else:
                    score_softRE = top / bottom



                #soft F1
                top = (score_softPR * score_softRE)
                bottom = (score_softPR + score_softRE)


                if top == 0 and bottom == 0:
                    score = 0.0
                elif top == 0:
                    score = 0.0
                else:
                    score = 2 * top / bottom


                scores.append(score)

            result_data_xaimethod_time['soft_F1_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['soft_F1_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)


        for version_soft in ["abs","pos","neg"]:


            # Soft Precision (AttributionLab)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0


            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy, explanations_copy):

                top = (abs(explanation * gt)).sum()
                if not top:
                    top = 0.0

                bottom = (abs(explanation)).sum()

                if top == 0 and bottom == 0:
                    score = 1.0
                elif top == 0:
                    score = 0.0
                else:
                    score = top / bottom

                scores.append(score)

            result_data_xaimethod_time['_soft_PR_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['_soft_PR_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)




            # Soft Recall (AttributionLab)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0

            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy, explanations_copy):

                top = (abs(explanation * gt)).sum()
                if not top:
                    top = 0.0

                bottom = (abs(gt)).sum()

                if top == 0 and bottom == 0:
                    score = 1.0
                elif top == 0:
                    score = 0.0
                else:
                    score = top / bottom

                scores.append(score)

            result_data_xaimethod_time['_soft_RE_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['_soft_RE_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)





            # Soft F1 (AttributionLab)

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')


            if version_soft == "abs":
                explanations_copy = np.abs(explanations_copy)
                gt_explanation_copy = np.abs(gt_explanation_copy)
            elif version_soft == "pos":
                explanations_copy[explanations_copy<0] = 0
                gt_explanation_copy[gt_explanation_copy<0] = 0
            elif version_soft == "neg":
                explanations_copy[explanations_copy>0] = 0
                gt_explanation_copy[gt_explanation_copy>0] = 0


            start_time = startTimer()
            scores = []
            for gt, explanation in zip(gt_explanation_copy, explanations_copy):


                # soft precision

                top = (abs(explanation * gt)).sum()
                if not top:
                    top = 0.0

                bottom = (abs(explanation)).sum()

                if top == 0 and bottom == 0:
                    score_softPR = 1.0
                elif top == 0:
                    score_softPR = 0.0
                else:
                    score_softPR = top / bottom



                # soft recall

                bottom = (abs(gt)).sum()

                if top == 0 and bottom == 0:
                    score_softRE = 1.0
                elif top == 0:
                    score_softRE = 0.0
                else:
                    score_softRE = top / bottom



                #soft F1
                top = (score_softPR * score_softRE)
                bottom = (score_softPR + score_softRE)


                if top == 0 and bottom == 0:
                    score = 0.0
                elif top == 0:
                    score = 0.0
                else:
                    score = 2 * top / bottom


                scores.append(score)

            result_data_xaimethod_time['_soft_F1_'+version_soft] = endTimer(start_time, num_items)
            result_data_xaimethod['_soft_F1_'+version_soft] = sum(scores) / len(scores)

            checkList(scores)






        # Attention Percentage Contributing (Do feature attribution methods correctly attribute features)
        print("Attention Percentage Contributing")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True) # False = contributing

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

            top = np.ma.array(abs(explanation), mask = gt).sum()
            if not top:
                top = 0.0

            bottom = abs(explanation).sum()

            if top == 0 and bottom == 0:
                score = 1.0
            elif top == 0:
                score = 0.0
            else:
                score = top / bottom

            scores.append(score)

        result_data_xaimethod_time['Atten_C'] = endTimer(start_time, num_items)
        result_data_xaimethod['Atten_C'] = sum(scores) / len(scores)

        checkList(scores)





        # Attention Percentage Contributing (Do feature attribution methods correctly attribute features)
        print("Attention Percentage Non-Contributing")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        gt_explanation_copy_binary = turn_binary(gt_explanation_copy, False) # False = Non-contributing

        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):

            top = np.ma.array(abs(explanation), mask = gt).sum()
            if not top:
                top = 0.0

            bottom = abs(explanation).sum()

            if top == 0 and bottom == 0:
                score = 1.0
            elif top == 0:
                score = 0.0
            else:
                score = top / bottom

            scores.append(score)

        result_data_xaimethod_time['Atten_NC'] = endTimer(start_time, num_items)
        result_data_xaimethod['Atten_NC'] = sum(scores) / len(scores)

        checkList(scores)


































        # Calc our metrics

        print("our metrics")

        for operator in ["!=",">","<"]:

            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')

            start_time = startTimer()
            compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
            result_data_xaimethod['cpa'+operator] = compactnessMetric.get_score()
            result_data_xaimethod_time['cpa'+operator] = endTimer(start_time, num_items)

            checkList(compactnessMetric._scores)


            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')

            start_time = startTimer()
            completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
            result_data_xaimethod['cpl'+operator] = completenessMetric.get_score()
            result_data_xaimethod_time['cpl'+operator] = endTimer(start_time, num_items)

            checkList(completenessMetric._scores)


            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')

            start_time = startTimer()
            compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
            completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
            correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
            result_data_xaimethod['cor'+operator] = correctnessMetric.get_score()
            result_data_xaimethod_time['cor'+operator] = endTimer(start_time, num_items)

            checkList(compactnessMetric._scores)
            checkList(completenessMetric._scores)


        #cor_nosign

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        start_time = startTimer()
        compactnessMetric = Our_CompactnessMetric( gt_explanation_copy,  explanations_copy,"!=")
        completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,"!=")
        correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
        result_data_xaimethod['cor_nosign'] = correctnessMetric.get_score()
        result_data_xaimethod_time['cor_nosign'] = endTimer(start_time, num_items)

        checkList(compactnessMetric._scores)
        checkList(completenessMetric._scores)


        #cor_sign

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')

        start_time = startTimer()
        compactnessMetric = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,">")
        completenessMetric = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,">")
        correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
        completenessscore_greater = correctnessMetric.get_score()

        compactnessMetric2 = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,"<")
        completenessMetric2 = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,"<")
        correctnessMetric2 = Our_CorrectnessMetric(completenessMetric2,compactnessMetric2)
        completenessscore_less = correctnessMetric2.get_score()

        result_data_xaimethod['cor_sign'] = (completenessscore_greater+completenessscore_less)/2.0
        result_data_xaimethod_time['cor_sign'] = endTimer(start_time, num_items)

        checkList(compactnessMetric._scores)
        checkList(completenessMetric._scores)

        checkList(compactnessMetric2._scores)
        checkList(completenessMetric2._scores)

        # Time metric

        result_data_xaimethod['Time'] = xai_method_time



        # Save images
        if settings['saveSelectedImages']:
            print("saving images...")
            selected_idx = ([0,3,1,4,2][conceptid])*settings['num_examples_per_class']
            class_idx = [0,3,1,4,2][conceptid]

            saveImageColomap(root_url+"paper_images/selected/True/"+str(settings['run_id'])+"/"+xai_name+"_"+str(class_idx)+".png", explanations[selected_idx])

            if xai_name == all_xai_methods_names[-1]:
              saveImageColomap(root_url+"paper_images/selected/True/"+str(settings['run_id'])+"/gt2d_"+str(class_idx)+".png", gt_explanation[selected_idx])
              saveImage(root_url+"paper_images/selected/True/"+str(settings['run_id'])+"/input_"+str(class_idx)+".png", data['test']['x'][selected_idx])


            if settings['saveAllImages']:

                saveImages(root_url+"paper_images/explanations/True/"+str(settings['run_id'])+"/",explanations,conceptid, True,data,"_"+xai_name) # (path,images,conceptid, applyColormap,data,postfix="")
                saveImages(root_url+"paper_images/input/True/"+str(settings['run_id'])+"/",data['test']['x'],conceptid, False,data)
                saveImages(root_url+"paper_images/input/True/"+str(settings['run_id'])+"/",gt_explanation,conceptid, True,data,"_gt")

            else:
                print("!!! NOT saving all images... !!!")
        else:
            print("!!! NOT saving images... !!!")

        # Store results

        if not xai_name in result_data:
            result_data[xai_name] = [result_data_xaimethod]
            result_data_time[xai_name] = [result_data_xaimethod_time]

        else:
            result_data[xai_name].append(result_data_xaimethod)
            result_data_time[xai_name].append(result_data_xaimethod_time)


        # ---------------------- test ---------------------- #

        if settings['normalize_explanations']:
            for exp in range(len(data['test']['y'])):

                if np.amax(explanations[exp]) > 1 or np.amin(explanations[exp]) < -1:
                    print("ERROR!")
                    exit()

                if np.amax(data['test']['x'][exp]) != 1 or np.amin(data['test']['x'][exp]) != 0:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation'][exp]) > 1 or np.amin(data['test']['gt_explanation'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) > 1 or np.amin(data['test']['gt_explanation3D'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) != np.amax(data['test']['gt_explanation'][exp]) or np.amin(data['test']['gt_explanation3D'][exp]) !=  np.amin(data['test']['gt_explanation'][exp]):
                    print("ERROR!")
                    exit()


result_data = {}
result_data_time = {}

def main():

    saveColormap()


    for conceptid in range(5):


        # ---------------------- Load Data ---------------------- #
        print("*** conceptid ",conceptid," ***")
        data = dataset_synth_newGT.generate(settings['num_examples_per_class'], conceptid)
        model_classifier = genModel(conceptid)


        # ---------------------- Evaluate model ---------------------- #
        loss = model_classifier.evaluate(data['test']['x'], data['test']['y'])
        if loss[0] > 0.0001:
            print("ERROR! Loss != 0")
            exit()

        # ---------------------- calc_metrics_for_XAI_method ---------------------- #

        for xai_method, xai_name in zip(all_xai_methods, all_xai_methods_names):
            calc_metrics_for_XAI_method(xai_method, xai_name, model_classifier, data, conceptid)


        # ---------------------- test ---------------------- #
        if settings['normalize_explanations']:
            for exp in range(len(data['test']['y'])):
                if np.amax(data['test']['x'][exp]) != 1 or np.amin(data['test']['x'][exp]) != 0:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation'][exp]) > 1 or np.amin(data['test']['gt_explanation'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) > 1 or np.amin(data['test']['gt_explanation3D'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) != np.amax(data['test']['gt_explanation'][exp]) or np.amin(data['test']['gt_explanation3D'][exp]) !=  np.amin(data['test']['gt_explanation'][exp]):
                    print("ERROR!")
                    exit()


    # ---------------------- Print and save results ---------------------- #

    for xai_name in all_xai_methods_names:
        result_data[xai_name] = dict_mean(result_data[xai_name])
        result_data_time[xai_name] = dict_mean(result_data_time[xai_name])

    df = pd.DataFrame(data=result_data)
    df.to_csv(root_url+'table_results_True_'+str(settings['run_id'])+'.csv', index = True)
    print(df.to_latex(index=True))

    print("")
    print("")
    print("")

    df_time = pd.DataFrame(data=result_data_time)
    #df_time = df_time.mean(axis=1)
    df_time.to_csv(root_url+'table_times_True_'+str(settings['run_id'])+'.csv', index = True)
    print(df_time.to_latex(index=True))


if __name__ == "__main__":

  all_xai_methods = [GradCAM,GradCAMPP,Saliency,DeconvNet, GradientInput, GuidedBackprop, IntegratedGradients, SmoothGrad, SquareGrad,VarGrad,Occlusion,Rise,KernelShap,Lime]
  all_xai_methods_names = ["GradCAM","GradCAMPP","Saliency","DeconvNet", "GradientInput", "GuidedBackprop", "IntegratedGradients", "SmoothGrad", "SquareGrad","VarGrad","Occlusion","Rise","KernelShap","Lime"]

  all_xai_metrics = [Deletion,Insertion, MuFidelity]
  all_xai_metrics_names = ["Deletion","Insertion", "MuFidelity"]


  for settings['run_id'] in range(3):

    result_data = {}
    result_data_time = {}

    print("******************* RUN: ",settings['run_id'],") *******************")

    main()

import pandas as pd
import numpy as np

folder = "Normalized-yes_Cuda-no"
num_runs = 3

for name in ['table_results_','table_times_']:

    dfs = []

    for i in range(num_runs):
        dfs.append(pd.read_csv(root_url+name+str(i)+'.csv', index_col=False))

    #print(dfs)
    #exit()

    df = pd.concat(dfs)
    df.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)

    by_row_index = df.groupby(df.Metric, sort=False)

    df_means = by_row_index.mean()

    #print(df_means)
    #exit()

    if name == 'table_times_':
        df_means = df_means.mean(axis=1)

    #print(df_means)
    #exit()

    df_std = by_row_index.std()

    #print(df_std)
    #exit()

    if name == 'table_times_':
        df_std = df_std.mean(axis=1)

    #print(df_std)
    #exit()

    df = df_means.round(2).astype(str) + u"\u00B1" + df_std.round(2).astype(str)

    #print(df)

    if name == 'table_times_':
        df = pd.DataFrame(df)

        df.insert(1, "Means", df_means, True)

        #print(df)
        #exit()

        df = df.sort_values(by=['Means'])

        #print(df)
        #exit()

        df =  df.drop(columns=['Means'])
        print(df.to_latex(index=True))

    else:

        #print(df)

        df['Delta'] = df_means.max(axis=1)-df_means.min(axis=1)
        df['Delta'] = df['Delta'].round(2).astype(str)

        #print(df)
        #exit()

        df = df.apply(lambda x: x.apply(lambda x: x.lstrip('0')))

        print(df.to_latex(index=True))


        #df = df.sort_values(by=['Delta'])

        #print(df.to_latex(index=True))