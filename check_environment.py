import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import google.protobuf as protobuf
import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("GPU Devices: ", tf.config.list_physical_devices('GPU'))
print("CUDA Version: ", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN Version: ", tf.sysconfig.get_build_info()['cudnn_version'])
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"protobuf version: {protobuf.__version__}")
print(f"Python version: {sys.version}")