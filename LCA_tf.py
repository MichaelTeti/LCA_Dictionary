import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from sklearn.preprocessing import scale
from scipy.misc import *
import os
import glob
from scipy import ndimage as ndi
from skimage import color
from skimage.util import view_as_windows as vaw
import h5py


learning_rate = 10
subsample=9
LCAiters=20
imsz=150
ps=16
training_iterations = 20001
num_classes=25
display_step = 20000
k=600 #number of receptive fields to learn
batch_sz=200 #batch_size


def read_ims(directory, imsz, whitening=False):
  ''' Reads in images in subdirectories located in directory and
      assigns a unique one-hot vector to each image in the respective
      folder.

      args:
           directory: the location of all the folders containing
                      each image class.
           imsz: resizes the width and height of each image to
                 imsz. '''

  main_dir=os.getcwd()
  os.chdir(directory)
  num_channels=1 #### remove

  num_ims=sum([len(files) for r, d, files in os.walk(directory)])
  imgs=np.zeros([num_ims, imsz, imsz, num_channels])
  labels=np.zeros([num_ims, len(os.listdir(os.getcwd()))])
  im_num=0
  class_num=0

  for f in os.listdir(os.getcwd()):
    if os.path.isdir(f):
        print('Folder name: %s'%(f))
        os.chdir(f)
        r0=np.argmin(np.sum(labels, axis=1))
        labels[r0:r0+len(glob.glob1(os.getcwd(), '*')), class_num]=1
        class_num+=1

        for filename in os.listdir(os.getcwd()):
          im=imresize(imread(filename), [imsz, imsz])
          im=im[:, :, np.newaxis]
          imgs[im_num, :, :, :]=im
          im_num+=1

        os.chdir(directory)
  os.chdir(main_dir)

  return imgs, labels


def LCA(x, D, num_iters):
    D=tf.matmul(D, tf.diag(1/(tf.sqrt(tf.reduce_sum(D**2, 0))+1e-6)))
    u=tf.zeros([k, batch_sz])

    for iters in range(num_iters):
        a=(u-tf.sign(u)*.0484)*(tf.cast(tf.abs(u)>.0484, tf.float32))
        u=u+.01*(tf.matmul(tf.transpose(D), (x-tf.matmul(D, a)))-u-a)

        #D=D+(1/batch_sz)*tf.matmul(x-tf.matmul(D, a), tf.transpose(a))
    return a


def montage(X):
    m, n, count = np.shape(X)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = X[:, :, image_id]
            image_id += 1
    return M


def mat2ten(X):
    zs=[int(np.sqrt(X.shape[0])),int(np.sqrt(X.shape[0])),X.shape[1]]
    Z=np.zeros(zs)
    for i in range(X.shape[1]):
        Z[:,:,i]=np.reshape(X[:,i],[zs[0],zs[1]])
    return Z


def plot(*args, **kwargs):
    fig=plt.figure()
    kwargs['plotsize']=kwargs.get('plotsize', [15, 15])
    plot_size=kwargs['plotsize']
    for i in range(len(args)):
        ax=fig.add_subplot(1, len(args), i+1)
        ax.imshow(args[i], cmap='gray')
    fig.set_size_inches(plot_size[0], plot_size[1])
    plt.show()


if os.path.isfile('texture_data.h5') is False:
    X=np.zeros([145*num_classes, imsz, imsz, 1])
    Y=np.zeros([145*num_classes])
    testx=np.zeros([15*num_classes, imsz, imsz, 1])
    testy=np.zeros([15*num_classes])

    x, y=read_ims('/home/voxelrx/Textures', imsz, whitening=False)
    x, y=img_aug(x, y, rotate=90, flip=True)
    #plot(y, plotsize=[200, 200])

    for i in range(num_classes):
        class_=(y[:, i]==1)
        class_ims=x[class_, :, :, :]
        X[i*145:i*145+145, :, :, :]=class_ims[:145, :, :, :]
        testx[i*15:i*15+15, :, :, :]=class_ims[145:, :, :, :]
        class_labels=y[class_, :]
        Y[i*145:i*145+145]=np.argmax(class_labels[:145, :], 1)
        testy[i*15:i*15+15]=np.argmax(class_labels[145:, :], 1)

    f=h5py.File('texture_data.h5', 'a')
    f.create_dataset('X', data=X)
    f.create_dataset('Y', data=Y)
    f.create_dataset('testx', data=testx)
    f.create_dataset('testy', data=testy)
    f.close()

else:
    f=h5py.File('texture_data.h5', 'r')
    X=np.asarray(f['X'])
    Y=f['Y']
    testx=np.asarray(f['testx'])
    testy=f['testy']

#Dicts=np.zeros([ps**2, k*num_classes])
x=tf.placeholder("float", [ps**2, None])
w=tf.Variable(tf.random_normal([ps**2, k]))
a=LCA(x, w, LCAiters)

cost=tf.sqrt(tf.reduce_sum(tf.abs(tf.matmul(w, a)-x)**2))+0.01*tf.reduce_sum(tf.abs(a))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


for class_num in range(num_classes):
    class_=(Y[:, class_num]==1)
    patches=vaw(X[class_, :, :, :], (1, ps, ps, X.shape[3]))
    patches=patches[:, ::subsample, ::subsample, :, :, :, :, :]
    patches=patches.reshape([patches.shape[0]*
                                patches.shape[1]*
                                patches.shape[2]*
                                patches.shape[3]*
                                patches.shape[4], -1]).transpose()
    patches=0.1*scale(patches, axis=0)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for iters in range(training_iterations):
            batch=patches[:, np.random.randint(0, patches.shape[1], size=batch_sz)]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch})

            if iters%display_step==0:
                print('Class: %d, Iteration: %d, Cost: %f'%(class_num, iters, c))
                wp=w.eval()
                plot(montage(mat2ten(wp)))
                A=sess.run(a, feed_dict={x:batch})
                A=np.sum(np.int32(A!=0.0), axis=0)
                plt.plot(np.arange(batch_sz), A)
                plt.show()

    Dicts[:, class_num*k:class_num*k+k]=wp
np.save('texture_dict.npy', wp)
