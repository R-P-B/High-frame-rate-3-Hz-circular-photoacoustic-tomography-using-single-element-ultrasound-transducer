import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from gooey import Gooey, GooeyParser
from matplotlib import pyplot as plt
from tensorflow.keras.layers import  Conv2D, Add, Input,BatchNormalization, Concatenate, Conv2DTranspose

@Gooey(program_name='Dataset preparation')
def main():
    setting_msg = 'Simulated Data '
    parser = GooeyParser(description=setting_msg)
    parser.add_argument('test_dir', help='test directory',type=str, widget='FileChooser',default='')
    parser.add_argument('model_dir', help='weights of the model', type=str, widget='FileChooser', default='')
    parser.add_argument('num_samples', help='number of samples to plot ',type=int,  default= 3)
    parser.add_argument('im_no', help='range of images ', type=int, default=0)
    args = parser.parse_args()
    return args.test_dir,args.model_dir,args.save_dir,args.num_samples,args.im_no
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

def model_load(test_filename,model_dir,num_samples,im_no):
    predictions_test = list()
    def data_gen(filename):
        path = filename;
        with np.load(path) as data:
            src_images = data['arr_0']
            tar_images = data['arr_1']
        return src_images, tar_images
    src_test , tar_test = data_gen(test_filename)
    print(len(src_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((src_test))
    test_dataset = test_dataset.batch(1)
    directory = os.path.join(os.getcwd(), model_dir)
    saved_model = getModel(input_shape=(128,128,1), filters=32, kernel_size=3, activation='relu')
    saved_model.load_weights(directory)
    saved_model.summary()
    print('Done Loading Model(' + model_dir + ') from: ' + model_dir)
    for element in test_dataset.as_numpy_iterator():
        predictions_curr = saved_model.predict(element, steps = 1)
        predictions_test.append(predictions_curr)
    [predictions_test] = [np.asarray(predictions_test)]
    predictions = np.reshape(predictions_test, (predictions_test.shape[0],128, 128))
    src_images = np.reshape(src_test, (src_test.shape[0],128, 128))
    tar_images = np.reshape(tar_test, (tar_test.shape[0],128, 128))
    for i in range(num_samples):
        plt.subplot(3, num_samples, 1 +  i)
        plt.axis('off')
        plt.imshow(src_images[i+im_no],cmap='gist_yarg')
        plt.title('input')
        plt.subplot(3, num_samples, 1 +num_samples+ i)
        plt.axis('off')
        plt.imshow(predictions[i+im_no],cmap='gist_yarg')
        plt.title('predicted')
        plt.subplot(3, num_samples, 1 + num_samples*2  + i)
        plt.axis('off')
        plt.imshow(tar_images[i+im_no],cmap='gist_yarg')
        plt.title('ground truth')
    plt.show()

if __name__=='__main__':
    test_dir, model_dir,num_samples, im_no = main()
    model_load(test_dir, model_dir,  num_samples, im_no)
