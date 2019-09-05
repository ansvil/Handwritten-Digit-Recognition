import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#test_labels= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#loading data
digit_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

#noising should be random for all the area
def noising(noise_lvl, noise):
    test_images_noise = test_images
    for elem in test_images_noise:
        sum=0
        while(sum<noise_lvl*28*28):
            elem[random.randint(0,27)][random.randint(0,27)]=noise
            sum=sum+1
    return   test_images_noise


#noising should be in a random line for each pic
def noising_line(noise):
    test_images_noise = test_images
    for elem in test_images_noise:
        j=random.randint(0,27)
        for i in range(0, len(elem[j])):
            elem[j][i]=noise
    return   test_images_noise

"""
#trying to load our data
path = 'C: \\Users\\svilpova.2015\\PycharmProjects\\test\\image_test'
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = datagen.flow_from_directory(
    directory = 'image_test',
    target_size=(28, 28),
    batch_size=1,
    class_mode='categorical')
"""
#normalizing images
train_images = train_images / 255.0
test_images = test_images / 255.0
#printing figures
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#test_images1 = noising(0.7,1)
test_images1 = noising_line(0.5)
plt.figure()
plt.imshow(test_images1[0])
plt.colorbar()
plt.grid(False)
plt.show()


#forming neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model training
model.fit(train_images, train_labels, epochs=5)

#testing accuracy of trained model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

def predict(test_image):
    #making predicitions on testing data
    predictions = model.predict(test_image)
    #the predicition to first elem of testing data compared to oit's actual label

    print("First elem of testing data is predicted to be: ", np.argmax(predictions[0]))
    print("Actual first elem is: ", test_labels[0])
    #total predicition accuracy
    sum=0
    for i in range(0,len(predictions)):
         if(np.argmax(predictions[i])==test_labels[i]):
             sum=sum+1

    print('Test noise accuracy: ', sum/len(predictions))
predict(test_images)
"""
#another way of predicting
scores = model.evaluate_generator(test_generator, 1 // 1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))



#noising the image with different noise lvls
noise_=[0, 0.3, 0.5, 0.7, 0.9]
for elem in noise_:
    noise_img=noising(elem)
    print("Noise level is: ", elem)
    predict(noise_img)
    print('----------------------------------------------------')"""
