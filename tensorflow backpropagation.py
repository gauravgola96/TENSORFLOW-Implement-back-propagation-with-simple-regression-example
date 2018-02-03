import tensorflow as tf

import numpy as np

tf.reset_default_graph()

sess = tf.Session()

x_vals = np.random.normal(loc=0.0,scale=0.1,size=100)
x_vals.shape

y_vals = np.repeat(10,100)

##create placeholder

X = tf.placeholder(dtype=tf.float32,shape=[1])

Y = tf.placeholder(dtype=tf.float32,shape=[1])

A = tf.Variable(tf.random_normal(shape=[1]))
#A is a variable 
#This means in optimization A will change 

output = tf.multiply(X,A)

error = tf.squared_difference(output,Y)

##error is calculated
## Now optimization 

#Before we can run anything, we have to initialize the variables
inti = tf.global_variables_initializer()

sess.run(inti)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)

train = optimizer.minimize(error)
#Optimizer is called 
                           

# # We will do this 101 times and print out results every 25th iteration.

#Now we have to call one value at a time from placeholders
## for loop (100)

for i in range(100):
    index = np.random.choice(100)
    X_random = [x_vals[index]]
    Y_random = [y_vals[index]]
    sess.run(train,feed_dict={X:X_random,Y:Y_random})
    if (i+1)%25==0:
        print("A" "=",sess.run(A))
        print("error = ",sess.run(error,feed_dict={X:X_random,Y:Y_random}) )

