import tensorflow as tf 
from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices() )

k=170
s=260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def lossf():
    w = k * a + b
    return tf.square(260 - w)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(lossf,var_list=[a,b])
    print(a.numpy(),b.numpy())




