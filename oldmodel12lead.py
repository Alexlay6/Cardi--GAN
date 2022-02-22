

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm



import matplotlib.pyplot as plt
import numpy as np

import pickle

import os

import time


# In[2]:


from IPython import display

print(tf.__version__)

print("GPU:", tf.test.is_gpu_available())
if tf.test.is_gpu_available():
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'


# In[3]:


BATCH_SIZE=64
def real_samples():  
    Z = np.zeros(shape = (1,7))
    X_real = np.loadtxt('X_CRBBB.csv')


    for i in range(1000):
        
        
        x=X_real[i,:]
        #x_z=(x-np.mean(x))/np.std(x)
        X_real[i,:]=(x-min(x))/(max(x)-min(x))
        
    
    X=X_real.reshape((3635,200,12))

    train_labels = np.ones((3635, 1))
    train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(10000).batch(BATCH_SIZE)
   
                                                  
    return train_dataset, train_labels, X
def real_samples_3():
    
    Z = np.zeros(shape = (1,7))
    X = np.empty((1180, 1600, 1))       
    for i in range(1180):
        with open("AFIB_sort/seg_{}.pkl".format(i), 'rb') as fx:
            X[i] = np.load(fx, allow_pickle = True) 
            X[i] = (X[i]*2)-1
            n = len(X)
       
    train_labels = np.ones((n, 1))
    
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(10000).batch(BATCH_SIZE)
    
                                                  
    return train_dataset, train_labels, X

train_dataset, train_labels, plot_data = real_samples()

# In[4]:


def real_samples_sine():  
    Z = np.zeros(shape = (1,7))
    X=np.zeros((1000,200))
    t=np.linspace(0,7,200)
    
    for i in range(1000):
        X[i,:]=np.sin(2*t)
        
    X=X.reshape((1000,200,1))
    train_labels = np.ones((1000, 1))
    train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(10000).batch(BATCH_SIZE)


    return train_dataset, train_labels, X

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[5]:





# In[6]:





# In[7]:




# In[16]:


def define_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=5, strides=2,padding = 'same', input_shape = (200, 12)))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



def define_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, use_bias=False, input_shape=(50,12)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((50, 12)))
    assert model.output_shape == (None, 50, 12)  # Note: None is the batch size

    model.add(layers.Conv1DTranspose(128, 5, strides=1, padding='same', use_bias=False))
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False))

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(12, 5, strides=2, padding='same', use_bias=False, activation='sigmoid'))
    #can change to tanh easily
 

    return model


# In[18]:


discriminator_model = define_discriminator()
discriminator_model.summary()
generator_model = define_generator()
generator_model.summary()


# In[19]:


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss
    print(total_loss)
    return total_loss


# In[20]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
                         
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)

fake_disc_accuracy = tf.keras.metrics.BinaryAccuracy('fake_disc_accuracy', threshold=0)
real_disc_accuracy = tf.keras.metrics.BinaryAccuracy('real_disc_accuracy', threshold=0)
fake_disc_accuracy_list, real_disc_accuracy_list = [], []


# In[21]:


EPOCHS = 3000

latent_dim = 12

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)


# In[26]:



number_epochs = []
gen_loss_ = []
disc_loss_ = []
all_losses = []
epoch_samples = []
lambda_gp = 10.0
def train(dataset, epochs):
    print('Training')
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        start = time.time()
        epoch_losses = []
        
        for beat in dataset:
            beat = tf.cast(beat, tf.float32)
            len_im = tf.shape(beat)[0]
            len_im = len_im.numpy()
            
            x_input = np.random.rand(latent_dim*50*len_im) #generate points in latent space
            noise = x_input.reshape(len_im,50,latent_dim)


            ######################################################################## 

            with tf.GradientTape() as gen_tape, tf.GradientTape() as d_tape:

                for i in range(7):
                    
                    generated = generator_model(noise, training=True)
             
                    real_output = discriminator_model(beat, training=True)

                    fake_output = discriminator_model(generated, training=True)

                gen_loss = -tf.math.reduce_mean(fake_output)

                d_loss_real = -tf.math.reduce_mean(real_output)
                d_loss_fake =  tf.math.reduce_mean(fake_output)
                d_loss = d_loss_real + d_loss_fake

                fake_disc_accuracy.update_state(tf.zeros_like(fake_output), fake_output)
                real_disc_accuracy.update_state(tf.ones_like(real_output), real_output)

                with tf.GradientTape() as gp_tape:

                    alpha = tf.random.uniform(
                                shape=[real_output.shape[0], 1, 1], 
                                minval=0.0, maxval=1.0)


                    interpolated = (alpha*beat + (1-alpha)*generated)

                    gp_tape.watch(interpolated)

                    d_critics_intp = discriminator_model(interpolated)


                    grads_intp = gp_tape.gradient(
                            d_critics_intp, [interpolated,])[0]

                    grads_intp_l2 = tf.sqrt(
                            tf.reduce_sum(tf.square(grads_intp), axis=[1, 2]))

                    grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))

                    d_loss = d_loss + lambda_gp*grad_penalty

                    d_grads = d_tape.gradient(d_loss, discriminator_model.trainable_variables)

                    d_optimizer.apply_gradients(
                        grads_and_vars=zip(d_grads, discriminator_model.trainable_variables))

                    g_grads = gen_tape.gradient(gen_loss, generator_model.trainable_variables)

                    generator_optimizer.apply_gradients(
                        grads_and_vars=zip(g_grads, generator_model.trainable_variables))

                    epoch_losses.append(
                          (gen_loss.numpy(), d_loss.numpy(), 
                               d_loss_real.numpy(), d_loss_fake.numpy()))
                    gen_loss_.append(gen_loss.numpy()) 
                    disc_loss_.append(d_loss.numpy())          
                    all_losses.append(epoch_losses)
                    number_epochs.append(epoch)


                    fake_disc_accuracy_list.append(fake_disc_accuracy.result().numpy())
                    real_disc_accuracy_list.append(real_disc_accuracy.result().numpy())

                    # fake_disc_accuracy.reset_states()
                    # real_disc_accuracy.reset_states()


                    
        print('Epoch {:-3d} | ET {:.2f} min | Avg Losses >>'
                    ' G/D {:6.2f}/{:6.2f} [D-Real: {:6.2f} D-Fake: {:6.2f}] | Accuracy: Real = {:6.2f} Fake = {:6.2f}'
                    .format(epoch, (time.time() - start_time)/60, 
                      *list(np.mean(all_losses[-1], axis=0)), real_disc_accuracy.result().numpy(), fake_disc_accuracy.result().numpy()))
                    

    # Save the model every 100 epochs
    
        if (epoch) % 25 == 0 or EPOCHS-epoch<50:
#             print('hhh')
            x_input = np.random.rand(latent_dim*50) #generate points in latent space
            noise = x_input.reshape(1,50,latent_dim)
            #checkpoint.save(file_prefix = checkpoint_prefix)
            predictions = generator_model(noise, training=False)
            
            
            x = predictions.numpy()

            plot_ex=plot_data[np.random.randint(0, 999)]

            plot_ex_tensor=tf.convert_to_tensor(plot_ex)

            
             #    generator_model.save('Saved Models/Norm LSTM/Bi-LSTM-Norm_epoch{}.h5'.format(epoch))
                

                                           


            
            c = np.array(range(200))
            c = c.reshape(200, 1)
         
           
            L1 = np.empty((200,1))
            L2 = np.empty((200,1))
            for i in range(200):
                y = x[:,i]
                L1[i] = y[:,0]
            
            L1 = L1.reshape(200,1)
                
            
            X = np.hstack((c, L1))
#             print(X)
      
            
            Y = np.hstack((c,plot_ex))
            fig, axs = plt.subplots(2)
            axs[0].plot(Y[:,0], Y[:,1], color = 'red', label='Real')
            axs[0].plot(X[:,0], X[:,1], color = 'blue', label='Fake')
            axs[0].legend(loc="upper right")
            axs[0].set_title('Epoch Number: {}'.format(epoch))
            axs[0].set_xlabel('Time interval')
            axs[0].set_ylabel('Normalised data value')
          
          
    
            axs[1].plot(number_epochs, gen_loss_, color = 'g', label = 'Gen loss')
            axs[1].plot(number_epochs, disc_loss_, color = 'c', label = 'Disc loss')
            axs[1].legend(loc="upper left")
            axs[1].set_xlabel('Epoch number')
            axs[1].set_ylabel('Loss')
            axs[1].grid()

           

            fig.savefig("Images_fo_old_12_Lead/Image_NORM_{}".format(epoch))
            plt.close(fig)
        if (epoch) % 250 == 0:
            x_input = np.random.rand(1000*50*12) #generate points in latent space
            noise = x_input.reshape(1000,50,latent_dim)
            predictions = generator_model(noise, training=False)
            x = predictions.numpy()
            x = x.reshape(200000,12)
          
            np.savetxt('GAN_DATA_OLD.csv', x)
            
    
      
      


# In[27]:


train(train_dataset, EPOCHS)


# In[ ]:





# In[ ]:




