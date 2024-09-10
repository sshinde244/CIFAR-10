CIFAR-10 Image Classification Program: Thoughts and Explanation

I began this problem by first trying to conceptualize how to use machine learning to
complete this problem, which is to categorize tens of thousands of images into 10 categories. I
considered the basic steps of machine learning:
1. Prep data
2. Choose a model
3. Train the model
4. Test the model through a neural network
5. Monitor results
This model is a generalization of the overall machine learning process, but it was whatI
was familiar with so I decided to follow it. To prep my data, I knew I had to load the CIFAR10
dataset onto my program without downloading every image. A quick google search led me to
Keras, a useful tool I continued to use throughout this project. With Keras, I was able to import the dataset onto my program using from keras.datasets import cifar10. At this point, I knew that I needed to somehow break down each image to pixels, so I utilized normalization, or essentially scaling pixels down to a value between 0 and 1 by dividing pixels max value by their minimum value, essentially dividing by 255.

In order to do this, we used the train and test variables for x and y using to_categorical, another tool from the Keras library. Now that the data was prepped, I needed to select a ML model for my program. The model that made the most sense for this was a Convolutional Neural Network, or a CNN. A CNN is most commonly used for analyzing visual imagery, making it perfect for my program. Essentially, it performs many filter operations, and uses the output as the input for the next operation. Performed at many different resolutions, it allows my program to more closely examine the image. I used the YouTube video by 3B1B* in order to learn more. For the CNN for my program, I tried multiple different parameters of the functions to match the specifications of the images from CIFAR10. 

Once the model was selected and prepared, I had to recursively train the model using epochs and a specified batch size. Initially, I set the number of epochs to 10. After a few tweaks,
my code ran correctly, and after 10 epochs my program had an accuracy of around 0.71, or close
to 71%. I wanted to increase the accuracy, so I increased the number of epochs to 15 and ran it
again. After this, the accuracy was 0.811. I altered the code again, adding a layer of convolution, after which my accuracy rose again to 0.8609.

Sources:
*Medium.com:
https://medium.com/@reddyyashu20/cnn-python-code-in-keras-and-pytorch-48680615d2d7
*3Blue1Brown YouTube:
https://youtu.be/KuXjwB4LzSA?si=9z7sK-Dyxxvm37Tl
