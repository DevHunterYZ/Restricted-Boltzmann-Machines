import numpy as np
import tensorflow as tf

%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display

import xrbm.models
import xrbm.train
import xrbm.losses
from xrbm.utils.vizutils import *

from tensorflow.examples.tutorials.mnist import input_data

data_sets = input_data.read_data_sets('MNIST_data', False)
training_data = data_sets.train.images

num_vis         = training_data[0].shape[0] #=784
num_hid         = 200
learning_rate   = 0.1
batch_size      = 100
training_epochs = 15

# Kodu yeniden çalıştırmak istediğimizde tensorflow grafını sıfırlayalım
tf.reset_default_graph ()

rbm = xrbm.models.RBM(num_vis=num_vis, num_hid=num_hid, name='rbm_mnist')

batch_idxs = np.random.permutation(range(len(training_data)))
n_batches  = len(batch_idxs) // batch_size

batch_data     = tf.placeholder(tf.float32, shape=(None, num_vis))
cdapproximator = xrbm.train.CDApproximator(learning_rate=learning_rate)
train_op       = cdapproximator.train(rbm, vis_data=batch_data)

reconstructed_data,_,_,_ = rbm.gibbs_sample_vhv(batch_data)
xentropy_rec_cost  = xrbm.losses.cross_entropy(batch_data, reconstructed_data)
# Önce şekil oluştur, böylece eğitim sırasında filtreleri çekmek için aynı şeyi kullanıyoruz.
fig = plt.figure(figsize=(12,8))

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch_i in range(n_batches):
            # Sadece minibatch veri miktarını alın..
            idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
            
            # Eğitim adımını çalıştırın.
            sess.run(train_op, feed_dict={batch_data: training_data[idxs_i]})
    
        reconstruction_cost = sess.run(xentropy_rec_cost, feed_dict={batch_data: training_data})

        W = rbm.W.eval().transpose()
        filters_grid = create_2d_filters_grid(W, filter_shape=(28,28), grid_size=(10, 20), grid_gap=(1,1))
        
        title = ('Epoch %i / %i | Reconstruction Cost = %f'%
                (epoch+1, training_epochs, reconstruction_cost))
        
        plt.title(title)
        plt.imshow(filters_grid, cmap='gray')
        display.clear_output(wait=True)
        display.display(fig)
