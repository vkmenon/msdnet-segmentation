import keras
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Cropping2D, Concatenate, BatchNormalization, Activation, Flatten
from keras.models import Model
from skimage import io

#############
# Functions #
#############
def dice_coef(y_true, y_pred, smooth = 1. ):
    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (tf.constant(2.) * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return coef

def mask_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return bce + 1 - dice

def get_batch(xdir="./data/train/x/",ydir="./data/train/y/",idx=0):
    images = os.listdir(xdir)
    images = sorted(images)
    x = io.imread(xdir+images[idx])
    x = np.expand_dims(x,axis=0)
    y = io.imread(ydir+'mask'+images[idx][3:])
    y = np.expand_dims(y,axis=0)
    yy = np.zeros((np.shape(y)[0],np.shape(y)[1],np.shape(y)[2],1))
    yy[:,:,:,0] = y
    
    return (x,yy)

print "Making Model"
#####################################################
# Define the Multi-Scale-Dense Segmentation Network #
#####################################################
layers = 4
scale_depth = 5
feature_maps = [32,32,64,64,128]
bottleneck = 16

encoding_layers = [ [ None for y in range( layers ) ] for x in range( scale_depth ) ]
input_image = Input(shape=(None,None,4))

##########################
# Create encoding layers #
##########################
x = BatchNormalization()(input_image)
x = Activation('relu')(x)
encoding_layers[0][0] = Conv2D(feature_maps[0], (3, 3), padding='same')(x)

for i in range(1,scale_depth):
    x = BatchNormalization()(encoding_layers[i-1][0])
    x = Activation('relu')(x)
    encoding_layers[i][0] = Conv2D(feature_maps[i], (3, 3), dilation_rate=2, padding='same')(x)

# Iteratively add appropriate layers and connections based on specified size of network
for scale in range(scale_depth):
    for layer in range(1,layers):
        # If at the first (original) scale, then there are no layers from a larger 
        # scale that need to be concatenated.
        if scale == 0:
            # layer 1 (2nd layer) layer only has one previous input, so no need to concatenate
            if layer == 1:
                x = BatchNormalization()(encoding_layers[scale][0])
            else:
                to_concatenate = [encoding_layers[scale][i] for i in (range(layer))]
                x = Concatenate()(to_concatenate)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x) 
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoding_layers[scale][layer] = Conv2D(feature_maps[scale], (3, 3), padding='same')(x)
        else:
            if layer == 1:
                x = BatchNormalization()(encoding_layers[scale-1][0])
            else:
                to_concatenate_prev = [encoding_layers[scale-1][i] for i in (range(layer))]
                x = Concatenate()(to_concatenate_prev)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            out_previous = Conv2D(feature_maps[scale], (3, 3), dilation_rate=2,padding='same')(x)

            if layer == 1:
                x = BatchNormalization()(encoding_layers[scale][0])
            else:
                to_concatenate = [encoding_layers[scale][i] for i in (range(layer))]
                x = Concatenate()(to_concatenate)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x) #TODO: bottleneck
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            out_current = Conv2D(feature_maps[scale], (3, 3), padding='same')(x)

            encoding_layers[scale][layer] = Concatenate()([out_previous, out_current])

##########################
# Create decoding layers #
##########################
decoding_layers = [ [ None for y in range( layers ) ] for x in range( scale_depth ) ]

for i in range(scale_depth):
    decoding_layers[i][0] = encoding_layers[i][-1]

# Very similar to encoding layer
for scale in range(scale_depth)[::-1]:
    for layer in range(1,layers):
        if scale == scale_depth-1:
            if layer == 1:
                x = BatchNormalization()(decoding_layers[scale][0])
            else:
                to_concatenate = [decoding_layers[scale][i] for i in (range(layer))] #TODO: pass as reference
                x = Concatenate()(to_concatenate)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x) #TODO: bottleneck
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            decoding_layers[scale][layer] = Conv2D(feature_maps[scale], (3, 3), padding='same')(x)
        else:
            if layer == 1:
                x = BatchNormalization()(decoding_layers[scale+1][0])
            else:
                to_concatenate_prev = [decoding_layers[scale+1][i] for i in (range(layer))]  #TODO: pass as reference
                x = Concatenate()(to_concatenate_prev)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x) #TODO: bottleneck
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            out_previous = Conv2D(feature_maps[scale], (3, 3),padding='same')(x)

            if layer == 1:
                x = BatchNormalization()(decoding_layers[scale][0])
            else:
                to_concatenate = [decoding_layers[scale][i] for i in (range(layer))] #TODO: pass as reference
                x = Concatenate()(to_concatenate)
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(bottleneck, (1, 1), padding='same')(x) #TODO: bottleneck
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            out_current = Conv2D(feature_maps[scale], (3, 3), padding='same')(x)

            decoding_layers[scale][layer] = Concatenate()([out_previous, out_current])

for i in range(scale_depth-1):
    x = BatchNormalization()(decoding_layers[i+1][-1])
    x = Activation('relu')(x)
    x = Conv2D(feature_maps[i], (3, 3), padding='same')(x)
    decoding_layers[i][-1] = Concatenate()([decoding_layers[i][-1],x])

x = BatchNormalization()(decoding_layers[0][-1])
x = Activation('relu')(x)
x = Conv2D(bottleneck, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# FINAL OUTPUT TENSOR (Mask of Predictions)
mask = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

model = Model(inputs=[input_image], outputs=[mask])

###################
# Train the model #
###################
epochs = 100
batch_size = 1

xdir="./data/train/x/"
ydir="./data/train/y/"

train_idx = np.arange(len(os.listdir(xdir))*600/670)
val_idx = np.arange(len(os.listdir(xdir))*600/670,len(os.listdir(xdir)))

print "Compiling Model"
model.compile(optimizer='adam', loss=mask_loss, metrics=[dice_coef])
print model.metrics_names

train_loss = 10^10
val_dice = 0

for i in range(epochs):
    
    best_train_loss = train_loss
    best_val_dice = val_dice
    
    np.random.shuffle(train_idx)
    #train (only works on batch size 1)
    print "Training - Epoch " + str(i)
    total_train_loss = []
    for j in train_idx:
        (x,y) = get_batch(xdir,ydir,j)
        l = model.train_on_batch(x,y)
        total_train_loss.append(l[0])
    train_loss = np.mean(total_train_loss)
    
    #eval on validation set
    print "Validating"
    total_val_loss = []
    for k in val_idx:
        (x,y) = get_batch(xdir,ydir,k)
        l = model.evaluate(x,y,verbose=0)
        total_val_dice.append(l[1])
    val_dice = np.mean(total_val_dice)
    
    print "Epoch: " + str(i) + ", Train Loss: " + str(train_loss) + ", Val Dice: " + str(val_dice)
    
    with open('msdlog.txt','a') as f:
        f.write("Epoch: " + str(i) + ", Train Loss: " + str(train_loss) + ", Val Dice: " + str(val_dice))
        
    if best_val_dice < val_dice:
        print "Validation improved! Saving model."
        model.save_weights('best_val.h5')
        print "Done saving."
        with open('msdlog.txt','a') as f:
            f.write("VALIDATION IMPROVED: " + str(best_val_dice))
        
    if best_train_loss > train_loss:
        print "Training improved! Saving model."
        model.save_weights('best_train.h5')
        print "Done saving."