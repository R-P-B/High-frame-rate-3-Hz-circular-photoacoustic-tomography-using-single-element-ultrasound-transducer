import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import  Conv2D,  Add, Input,BatchNormalization, Concatenate, Conv2DTranspose

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
########################################################################################################################
''' Hyper parameters'''
########################################################################################################################
B1 = 1
B2 = 0.01
EPOCHS = 100
FILTERS = 32
BATCH_SIZE = 2
KERNEL_SIZE = 3
INITIAL_LR = 0.001
ACTIVATION = 'relu'
delete_previous=True
model_dir = 'HD_UNet_optimized'
STANDARD_IMAGE_SHAPE = (128,128,1)
#########################################################################
''' Choose the training file name'''
#########################################################################

train_filename = '**********\train_set.npz'

#########################################################################
''' Choose the validation file name'''
#########################################################################

valid_filename = '**********\valid_set.npz'



########################################################################################################################
''' HD-UNet'''
########################################################################################################################
##########################################################################
'''MODEL FUNCTIONS:'''
##########################################################################
def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    ############################################
    out = DD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    shortcut = out
    out = DownSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return [out, shortcut]
###########################################################################
def RES(input, filters,kernel_size, padding, activation, kernel_initializer):
    ############################################
    f_in = filters
    out = input
    shortcut = out
    out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    out = Add()([out, shortcut])
    out = UpSample(out, f_in*2, kernel_size, strides=2, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return out
###########################################################################
def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    ############################################
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    out = UpSample(out, filters , kernel_size, strides=2, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return out
################################################################################
'''SUBFUNCTIONS FOR FUNCTIONS:'''
################################################################################
def Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding='same',
                     dilation = (1,1),activation='linear', kernel_initializer='glorot_normal'):
    ############################################
    out = Conv2D(filters=filters, kernel_size=kernel_size,
                 strides=strides, padding=padding,dilation_rate= dilation,
                 activation=activation,
                 kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    ############################################
    return out
###########################################################################
def Conv2D_Transpose_BatchNorm(input, filters, kernel_size=3, strides=2, padding='same',
                               dilation = (1,1),activation='relu', kernel_initializer='glorot_normal'):
    ############################################
    out = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,dilation_rate= dilation,
                          activation=activation, kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    ############################################
    return out
###########################################################################
def DownSample(input, filters, kernel_size=3, strides=2, padding='same',
               activation='linear', kernel_initializer='glorot_normal'):
    ############################################
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=strides, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return out
###########################################################################
def UpSample(input, filters, kernel_size=3, strides=2, padding='same',
             activation='linear', kernel_initializer='glorot_normal'):
    ############################################
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_Transpose_BatchNorm(out, filters // 2, kernel_size, strides=strides, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return out
##################################################################################
''' DILATED DENSE BLOCK'''
##################################################################################
def DD_Block(input, f_in, f_out, k, kernel_size=3, padding='same',activation='linear', kernel_initializer='glorot_normal'):
    out = input
    for i in range(f_in, f_out, k):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out_a = Conv2D_BatchNorm(out, filters=k/2, kernel_size=kernel_size, strides=1, padding=padding,dilation=(1,1),
                               activation=activation, kernel_initializer=kernel_initializer)
        out_b = Conv2D_BatchNorm(out, filters=k/2, kernel_size=kernel_size, strides=1, padding=padding,dilation=(2,2),
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Concatenate()([out_a,out_b, shortcut])
    return out
################################################################################
'''FULLY DENSE BLOCK:'''
################################################################################
def FD_Block(input, f_in, f_out, k, kernel_size=3, padding='same',
             activation='linear', kernel_initializer='glorot_normal'):
    out = input
    for i in range(f_in, f_out, k):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Conv2D_BatchNorm(out, filters=k, kernel_size=kernel_size, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Concatenate()([out, shortcut])
    return out
##################################################################################
''' HYBRID DENSE UNet'''
##################################################################################
def H_D_UNet(input, filters=32,no_of_resnetblocks =1, kernel_size=3, padding='same',activation='relu', kernel_initializer='glorot_normal'):
    shortcut1_1 = input
    out = Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    [out, shortcut1_2] = DownBlock(out, filters * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut2_1] = DownBlock(out, filters * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut3_1] = DownBlock(out, filters * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut4_1] = DownBlock(out, filters * 2 * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = RES(out, filters * 2 * 2 * 2 * 2,  kernel_size, padding, activation,kernel_initializer)
    out = Concatenate()([out, shortcut4_1])
    out = UpBlock(out, filters * 2 * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut3_1])
    out = UpBlock(out, filters * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut2_1])
    out = UpBlock(out, filters * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut1_2])
    out = FD_Block(out, f_in=filters, f_out=filters * 2, k=filters // 4, kernel_size=3, padding='same',
                   activation='linear', kernel_initializer='glorot_normal')
    out = Conv2D(filters=1, kernel_size=1, strides=1, padding=padding, activation='linear',
                 kernel_initializer=kernel_initializer)(out)
    out = Add()([out, shortcut1_1])
    return out

def getModel(input_shape, filters, kernel_size, padding='same',activation='relu', kernel_initializer='glorot_normal'):
    model_inputs = Input(shape=input_shape, name='img')
    model_outputs = H_D_UNet(model_inputs, filters=filters, kernel_size=kernel_size, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    model = Model(model_inputs, model_outputs, name='HD-UNet_Model')
    return model
########################################################################################################################
'''DATA LOADER'''
########################################################################################################################
if not os.path.exists(model_dir):
            os.mkdir(model_dir)
elif delete_previous:
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
bestmodel_dir = model_dir
if not os.path.exists(bestmodel_dir):
            os.mkdir(bestmodel_dir)
elif delete_previous:
            shutil.rmtree(bestmodel_dir)
            os.mkdir(bestmodel_dir)
def data_gen(filename):
    path = filename;
    with np.load(path) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images
src_train , tar_train = data_gen(train_filename)
src_valid , tar_valid = data_gen(valid_filename)
print(len(src_train)),print(len(src_valid))
train_dataset = tf.data.Dataset.from_tensor_slices((src_train, tar_train))
train_dataset = train_dataset.repeat(-1)
valid_dataset = tf.data.Dataset.from_tensor_slices((src_valid, tar_valid))
valid_dataset = valid_dataset.repeat(-1)
train_dataset = train_dataset.batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)

########################################################################################################################
'''MODEL UTILS'''
########################################################################################################################
def FFT_mag(input):
    ############################################
    real = input
    imag = tf.zeros_like(input)
    out = tf.abs(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0]))
    ############################################
    return out
###########################################################################
def model_loss(B1=1.0, B2=0.01):
    ############################################
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)
        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss
        return loss
    ############################################
    return loss_func
###########################################################################
def normalize(tensor):
    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))
###########################################################################
def PSNR(y_true, y_pred):
    ############################################
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    ############################################
    return PSNR
###########################################################################
def SSIM(y_true, y_pred):
    ############################################
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)
    ############################################
    return SSIM
###########################################################################
def KLDivergence(y_true, y_pred):
    return tf.losses.KLDivergence()(y_true, y_pred)
###########################################################################
def SavingMetric(y_true, y_pred):
    ############################################
    ssim = SSIM(y_true, y_pred)
    psnr = PSNR(y_true, y_pred)
    SSIM_norm = 1 - ssim
    PSNR_norm = (40 - psnr)/275
    loss = SSIM_norm + PSNR_norm
    ############################################
    return loss
########################################################################################################################
'''DATA LOADER'''
########################################################################################################################
opt = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
unet_model = getModel(input_shape=STANDARD_IMAGE_SHAPE, filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION)
unet_model.compile(optimizer=opt,  loss=model_loss(B1,B2), metrics=['mean_absolute_error', 'mean_squared_error', KLDivergence, SavingMetric, PSNR, SSIM]),
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
bestmodel_callbacks = ModelCheckpoint(filepath = os.path.join(bestmodel_dir, 'saved_model.epoch_{epoch:02d}-SSIM_{val_SSIM:.5f}-PSNR_{val_PSNR:.5f}-metric_{val_SavingMetric:.5f}.h5'), monitor='val_SavingMetric', verbose=0, save_best_only=False, save_weights_only=True, mode='min', save_freq='epoch')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_SavingMetric', factor=0.5, patience=5, verbose=1, mode='min',min_lr=0.000001,epsilon = 1e-04,)
history = unet_model.fit(train_dataset,  steps_per_epoch=np.ceil(len(src_train)/BATCH_SIZE),epochs=EPOCHS, callbacks=[bestmodel_callbacks,  reduce_lr_loss],validation_data=valid_dataset,validation_steps=np.ceil(len(src_valid)/BATCH_SIZE), max_queue_size = 256, shuffle = True)
########################################################################################################################
