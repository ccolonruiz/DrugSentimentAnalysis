# This module is used for general utils, such as the start of the environment, special plots and so on.
import os
import shutil

def start_environment():
    """
    This function is used to allocate memory dinamically in the GPU.
    """
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.backend import set_session
  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    
    np.random.seed(1)
    tf.set_random_seed(2)
    
def check_dir(d):
    """
    This function is in charge of checking if a directory exists.
    
    If not, said directory is created.
    """
    if not os.path.isdir(d):
        os.makedirs(d)
        print("Created dir:\n{}".format(d))
        
def del_dir(d):
    """
    This function is in charge of checking if a directory exists.
    
    If so, that directory is deleted.
    """
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print("Deleted dir:\n{}".format(d))