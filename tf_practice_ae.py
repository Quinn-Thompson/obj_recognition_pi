import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle5 as pickle
import cv2
import random
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle

from tensorflow.keras import layers, losses, optimizers, Input
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence as seq
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow import convert_to_tensor
import keras.backend as K


import DBM

class DataGenerator(seq):
    '''
        this class is handled using a lot of function overloading, I believe
        determination of how the generation will parse batches
        essentially, pass the contents of a folder (specifically a list of file names)
        it will then 
        Heavily influenced by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        by Afshine Amidi and Shervine Amidi
    '''
    def __init__(self, figure_path, label_path, file_names, num_classes, dimensions=(224,224), batch_size=16,
                n_channels=1, shuffle=True):
        '''
            just the initialization of each of the variables used in the class
        '''
        # dimensions of the data (1d? 2d? 3? size?)
        self.dimensions = dimensions 
        # size of each of the batches that the NN will parse
        self.batch_size = batch_size 
        # number of channels the NN will use (color? 3d?)
        self.n_channels = n_channels
        # shuffle data
        self.shuffle = shuffle 
        # path for figures
        self.figure_path = figure_path
        # file names in the figure path, shared between labels and figures
        self.file_names = file_names
        # path to the lables
        self.label_path = label_path
        # number of classes in the output layer
        self.num_classes = num_classes
        
        self.on_epoch_end()
    
    def __len__(self):
        '''
            parsed when the NN queries for the length of the actual batch
        '''
        # number of batches per epoch
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        '''
            parses the index of the epoch to return a set of data back to the NN
        '''
        # sfni = shuffled file names indeces
        # slices the total shuffled file list into a set of indeces that correspond to the current batch
        # this approach is taken for iterator shuffling as it does not manipulate the base data
        sfni_iterated_slice = self.shuffled_file_names_indeces[index*self.batch_size:(index+1)*self.batch_size]
        # this obtains the file names based on the passed batch of indexes
        shuffled_file_names_batch  = [self.file_names[x] for x in sfni_iterated_slice]
        # obtain the figures and the labels
        figures, labels = self.acquire_data(shuffled_file_names_batch)

        return figures, labels

    def on_epoch_end(self):
        '''
            at the end of each epoch and after the definition
        '''
        # arrange a list of indeces (0, 1, 2 ..., n) of the length of the number of files
        self.shuffled_file_names_indeces = np.arange(len(self.file_names))
        if self.shuffle == True:
            # randomly shuffle the indeces
            np.random.shuffle(self.shuffled_file_names_indeces)
    
    def acquire_data(self, shuffled_file_names_batch):
        # loaded array as placeholder for batching the CNN
        loaded_figures = np.empty((self.batch_size, *self.dimensions, self.n_channels))  
        loaded_labels = np.empty((self.batch_size, self.num_classes))  

        # for each file in the file batch
        for i, file_name in enumerate(shuffled_file_names_batch):
            # load the file 
            with open(self.figure_path + file_name, "rb") as fd:
              loaded_figure = pickle.load(fd)
            with open(self.label_path + file_name, "rb") as fd:
              loaded_label = pickle.load(fd)

            # bring it into the batch. This line is an artefact from when there was distinct operations between the pick load and the batch 
            # input label is 1344 length, which is 8x8 (grid) * 3 (number of bboxes available) * 7 (1 probability neuron, 4 bbox neurons, 2 class neurons)
            loaded_figures[i,] = loaded_figure
            loaded_labels[i] = loaded_label

        return loaded_figures, loaded_labels


# why is this distinct? residual addition should happen before activation and normalization
def conv_layer_res_end(prev_layer,
                       residual,
                       filter_size=16, 
                       kernel_size=(3,3),
                       pad_type='same',
                       activation='relu',
                       dropout=0.0):

  x = layers.Conv2D(filter_size, kernel_size, padding=pad_type)(prev_layer) # densely connects x neurons with the flat image

  x = layers.Add()([x, residual])
  # normalize values between layers so that extremes do not cause rampant exponential increase in weights.
  # this is a more likely case when the activations are monotonic like relu
  # The normalization limits weight increasing and exact weight fitting
  x = layers.BatchNormalization()(x)
  
  # dropout applied before activation as to force dropped neurons into the less computationally expensive 
  # half when activation occurs (x=0 as opposed to x=x)
  if dropout != 0.0:
    if activation == 'relu':
      x = layers.Dropout(dropout)(x)

  output_conv2d = layers.Activation(activation)(x)
  
  if dropout != 0.0:
    if activation != 'relu':
      output_conv2d = layers.Dropout(dropout)(output_conv2d)  

  return output_conv2d

#convolutional layer
def conv_layer(prev_layer,
              filter_size=16, 
              kernel_size=(3,3), 
              pad_type='same',
              activation='relu',
              dropout = 0.0,
              strides = (1,1)):
  
  # simple conv2d layer
  x = layers.Conv2D(filter_size, kernel_size, padding=pad_type, strides=strides)(prev_layer) 
  # normalize values between layers so that extremes do not cause rampant exponential increase in weights.
  # this is a more likely case when the activations are monotonic like relu
  # The normalization limits weight increasing and exact weight fitting
  x = layers.BatchNormalization()(x)

  # dropout applied before activation as to force dropped neurons into the less computationally expensive 
  # half when activation occurs (x=0 as opposed to x=x)
  if dropout != 0.0:
    if activation == 'relu':
      x = layers.Dropout(dropout)(x)

  output_conv2d = layers.Activation(activation)(x)
  
  # otherwise, activate afterwards, as either side includes an equation
  if dropout != 0.0:
    if activation != 'relu':
      output_conv2d = layers.Dropout(dropout)(output_conv2d)  

  return output_conv2d

# sequence layer for a resnet
def resnet_seq_layer(prev_layer, 
                     filter_size=16, 
                     kernel_size=(3,3), 
                     dropout=0.0,
                     stacked=False,
                     activation='relu'):
  # convolutional layer, including batch norm and activation
  x = conv_layer(prev_layer=prev_layer, filter_size=filter_size, kernel_size=kernel_size)

  # if there is another resnet sequence stacked on to this one, or just a previous conv layer with the same filter size
  # we can keep the residual daisy chain going
  if stacked == True:
    residual_x = conv_layer_res_end(prev_layer=x, residual=prev_layer, filter_size=filter_size, kernel_size=kernel_size, dropout=dropout)
  # stay with the normal conv layer
  else:
    residual_x = conv_layer(prev_layer=x, filter_size=filter_size, kernel_size=kernel_size, dropout=dropout)

  x = conv_layer(prev_layer=residual_x, filter_size=filter_size, kernel_size=kernel_size)
  # output conv layer is distinct as it includes an addition before batch normalization, a common aspect of some resnets
  output_resnet_seq = conv_layer_res_end(prev_layer=x, residual=residual_x, filter_size=filter_size, kernel_size=kernel_size)

  return output_resnet_seq

# Mean Squared Error of the probability of known non-object grids 
def KNOG_PMSE(label_actual, label_pred):
  label_obj_inverse = (label_actual[:,0::7] + 1) % 2 
  return K.sum(label_obj_inverse * (K.square(label_pred[:,0::7] - label_actual[:,0::7]))) / K.sum(label_obj_inverse)

# Mean Squared Error of the probability of known object grids 
def KOG_PMSE(label_actual, label_pred):
  return K.sum(label_actual[:,0::7] * (K.square(label_pred[:,0::7] - label_actual[:,0::7]))) / K.sum(label_actual[:,0::7])

# magnitude between the two center points of the bounding boxes for known object grids 
def KOG_PMAG(label_actual, label_pred):
  return K.sum( ( label_actual[:,0::7] ) * K.sqrt( K.square( label_pred[:,1::7] - label_actual[:,1::7] ) + K.square( label_pred[:,2::7] - label_actual[:,2::7] ) ) ) / K.sum(label_pred[:,0::7])

# difference in the sqrt of the area of the two bounding boxes for known object grids
def KOG_AREA_1DDIF(label_actual, label_pred):
  return K.sum( (label_actual[:,0::7] ) * K.abs( K.sqrt( (label_pred[:,3::7] * label_pred[:,4::7]) - (label_actual[:,3::7] * label_actual[:,4::7]) ) ) ) / K.sum(label_pred[:,0::7])

# summed difference in object class for known object grids 
def KOG_OBJ_MSE(label_actual, label_pred):
  return K.sum( ( label_actual[:,0::7] ) * K.square(label_pred[:,5::7] - label_actual[:,5::7] + label_pred[:,6::7] - label_actual[:,6::7] ) ) / K.sum(label_pred[:,0::7])

# custom you only look once v1 
# don't know why everyone uses Y and X for machine learning, makes everything hella confusing
# as opposed to labels and figures, which are unique terms
def loss():
  def custom_yolo_loss(label_actual, label_pred):
     # this could definetely be made more effecient and less explicit
     # I was just struggling to put together the complex loss, and I decided to make it extremely explicit to help myself
     
     # weight added to coordinates, as summed together they have less weight than probability and classes
     lambda_coor = 5
     # weight removed from no obj, as we don't want extreme correction for when known non-object grids are not close to zero
     # the correction is more reserved for when a known object is in a grid, as there are much less of those per image usually
     lambda_noobj = 0.5
     
     # penalize the x and y location, only when the 0th neuron is 1, based on squared difference, similair to magnitude 
     penalize_localality_x = lambda_coor*K.sum(label_actual[:,0::7]*(K.square(label_pred[:,1::7] - label_actual[:,1::7])))
     penalize_localality_y = lambda_coor*K.sum(label_actual[:,0::7]*(K.square(label_pred[:,2::7] - label_actual[:,2::7])))
     # penalize the height and width, only when the 0th neuron is 1, but add more weight to object thats are smaller, thus we use the square root
     penalize_localality_w = lambda_coor*K.sum(label_actual[:,0::7]*(K.square(K.sqrt(label_pred[:,3::7]) - K.sqrt(label_actual[:,3::7]))))
     penalize_localality_h = lambda_coor*K.sum(label_actual[:,0::7]*(K.square(K.sqrt(label_pred[:,4::7]) - K.sqrt(label_actual[:,4::7]))))
     
     # adjust probablitiy if object exists, only when the 0th neuron is 1
     penalize_probability_obj = K.sum(label_actual[:,0::7]*(K.square(label_pred[:,0::7] - label_actual[:,0::7]))) 
     # inverse the probability neuron
     label_obj_inverse = (label_actual[:,0::7] + 1) % 2 
     # penalize when the probability neuron is not close to 0, when a known non-object is in the grid
     # makes sure to adjust the probability if no object exists at 0.5 rate, as there are many more instances where this is true
     penalize_probability_noobj = lambda_noobj*K.sum(label_obj_inverse*(K.square(label_pred[:,0::7] - label_actual[:,0::7])))
     # penalize when the class is wrong
     penalize_class_inaccuracy0 = K.sum(label_actual[:,0::7]*(K.square(label_pred[:,5::7] - label_actual[:,5::7]))) 
     penalize_class_inaccuracy1 = K.sum(label_actual[:,0::7]*(K.square(label_pred[:,6::7] - label_actual[:,6::7]))) 
     # sum together for a single loss value, allowing for back propogation
     return (penalize_localality_x + penalize_localality_y + penalize_localality_w + 
                                     penalize_localality_h + penalize_probability_noobj + 
                                     penalize_class_inaccuracy0 + penalize_class_inaccuracy1
                                     + penalize_probability_obj)
  return custom_yolo_loss

if __name__ == '__main__':

  # from tensorflow.keras import layers
  if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF")


  label_path = 'database\\label_w_shift\\'
  figure_path = 'database\\figure_w_shift\\'

  val_split = 0.7
  test_split = 0.9

  # load file names from folder
  file_names = os.listdir(figure_path)
  # shuffle them for the first time so that when val and train are split, they get similair objects 
  # (comment out for a more accurate vali accuracy, but worse accuracy in general)
  file_array_shuffle = random.sample( file_names, len(file_names) )
  # split training and validation files
  training_files=file_names[:int(len(file_array_shuffle)*0.90)]
  validation_files=file_names[int(len(file_array_shuffle)*0.90):]
  
  #####################################
  # Our Model
  #####################################

  resnet_input = Input(shape=(64,64,3), name='img')
  main_path = conv_layer(prev_layer=resnet_input, filter_size=64, kernel_size=(7,7))
  main_path = layers.MaxPool2D(2,2)(main_path)
  main_path = layers.Dropout(0.2)(main_path)
  main_path = resnet_seq_layer(main_path, 16, (3, 3))
  main_path = layers.MaxPool2D(2,2)(main_path)
  main_path = layers.Dropout(0.2)(main_path)
  
  path_f = conv_layer(prev_layer=main_path, filter_size=32, kernel_size=(5,5), strides=(2,2))
  path_f = layers.MaxPool2D(2,2)(path_f)
  path_f = layers.Dropout(0.2)(path_f)
  path_f = conv_layer(prev_layer=path_f, filter_size=64, kernel_size=(5,5), strides=(2,2))
  path_f = layers.MaxPool2D(2,2)(path_f)

  path_f = layers.Flatten()(path_f) # flatten it into 1D neuron layer

  path_f = layers.Dense(256)(path_f)
  path_f = layers.Activation('relu')(path_f) 
  resnet_output_obj = layers.Dense((8*8*3*7))(path_f) 

  #####################################
  # Our Model End
  #####################################

  # optimizer Adam, cause it's the most commonly used
  optimizer = optimizers.Adam(lr=0.001)

  # if model doesn't exist yet
  if len(os.listdir('database\\my_model\\')) == 0:
    # compile the model into a single object
    resnet = Model(inputs=resnet_input, outputs=resnet_output_obj)
  
    # summarize the model in text
    resnet.summary()

    # generate a neat plot of the model
    plot_model(resnet, "database/resnet.png", show_shapes=True)


    training_generator = DataGenerator(figure_path=figure_path, label_path=label_path, file_names=file_names, dimensions=(64,64), batch_size=32, n_channels=3, num_classes=1344, shuffle=True)
    validation_generator = DataGenerator(figure_path=figure_path, label_path=label_path, file_names=file_names, dimensions=(64,64), batch_size=32, n_channels=3, num_classes=1344, shuffle=True)


    
    #resnet.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=['MeanAbsoluteError', 'MeanAbsolutePercentageError'])
    resnet.compile(
        optimizer=optimizer, 
        loss=loss(),
        metrics=[KNOG_PMSE, KOG_PMSE, KOG_PMAG, KOG_AREA_1DDIF, KOG_OBJ_MSE]
    )

    # save checkpoint when val loss is lowest
    checkpoint_filepath = 'database/checkpoints/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # fit the resnet using our training and validation generators to pull from hard drive
    # save checkpoint for best validation loss
    resnet.fit(training_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    verbose=2,
                    callbacks= [model_checkpoint_callback])
      
    # load weights back from check point
    resnet.load_weights(checkpoint_filepath)

    # set these to none so the model doesn't compile on load, throwing error
    resnet.optimizer = None
    resnet.compiled_loss = None
    resnet.compiled_metrics = None

    # save model
    resnet.save('database\\my_model\\my_model')
  else:
    resnet = load_model('database\\my_model\\my_model')
    print('model already created')

