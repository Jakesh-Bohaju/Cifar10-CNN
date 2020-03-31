import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        initializer = tf.initializers.GlorotUniform(seed=123)
        # Conv1
        self.wc1 = tf.Variable(initializer([3, 3, 3, 10]), trainable=True, name='wc1')
        
        # Conv2
        self.wc2 = tf.Variable(initializer([3, 3, 10, 20]), trainable=True, name='wc2')
        
        # Conv3
        self.wc3 = tf.Variable(initializer([3, 3, 20, 40]), trainable=True, name='wc3')
        
        # Flatten
        
        # Dense
        self.wd3 = tf.Variable(initializer([640, 280]), trainable=True)
        self.wd4 = tf.Variable(initializer([280, 80]), trainable=True)        
        self.wd5 = tf.Variable(initializer([80, 10]), trainable=True)
        
        self.bc1 = tf.Variable(tf.zeros([10]), dtype=tf.float32, trainable=True)
        self.bc2 = tf.Variable(tf.zeros([20]), dtype=tf.float32, trainable=True)
        self.bc3 = tf.Variable(tf.zeros([40]), dtype=tf.float32, trainable=True)
        
        self.bd3 = tf.Variable(tf.zeros([280]), dtype=tf.float32, trainable=True)
        self.bd4 = tf.Variable(tf.zeros([80]), dtype=tf.float32, trainable=True)        
        self.bd5 = tf.Variable(tf.zeros([10]), dtype=tf.float32, trainable=True)   
    
    def call(self, x):
        # X = NHWC 
        # Conv1 + maxpool 2
        x = tf.nn.conv2d(x, self.wc1, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bc1)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        # Conv2 + maxpool 2
        x = tf.nn.conv2d(x, self.wc2, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bc2)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        # Conv3 + maxpool 3
        x = tf.nn.conv2d(x, self.wc3, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bc3)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        # Flattten out
        # N X Number of Nodes
        # Flatten()
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        
        # Dense1
        x = tf.matmul(x, self.wd3)
        x = tf.nn.bias_add(x, self.bd3)
        x = tf.nn.relu(x)

        
        # Dense2
        x = tf.matmul(x, self.wd4)
        x = tf.nn.bias_add(x, self.bd4)
        x = tf.nn.relu(x)
        
        
        # Dense3
        x = tf.matmul(x, self.wd5)
        x = tf.nn.bias_add(x, self.bd5)
#         x = tf.nn.sigmoid(x)
        
        return x

def preprocess(image):
    image = tf.expand_dims(image, axis=0)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (28, 28))
    image = tf.dtypes.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image


def predict_top_1(predictions):
    # model = tf.saved_model.load('../temp/models/')
    return tf.argmax(tf.nn.softmax(predictions), axis=1)

def predict_top_3(predictions):
    outputs = tf.math.top_k(tf.nn.softmax(predictions), k=3)

    top_k = []
    for confidences, indices in zip(outputs[0].numpy(), outputs[1].numpy()):
        single_sample = dict()
        for confidence, index in zip(confidences, indices):
            single_sample[index] = confidence
        top_k.append(single_sample)
    return top_k


if __name__ == "__main__":
    model = LeNet()

    model_path = os.path.join(BASE_DIR, 'models', 'weights.h5')
    print(model_path)
    model.load_weights(model_path)
