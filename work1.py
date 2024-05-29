import tensorflow as tf 
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices() )


tensor = tf.constant([3.0,4,5])
tensor2 = tf.constant([[1,2],[3,4]])

print(tensor)

w=tf.Variable(1.0)
w.assign(2)
print(w)
