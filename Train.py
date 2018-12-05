import tensorflow as tf
import keras
import os
import  json
import math
import numpy as np
import h5py
from tensorflow.examples.tutorials.mnist import input_data
'''
mnist = input_data.read_data_sets("", one_hot=True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500

n_classes = 10
batch_size = 100

def neural_network_model(data):
    #setup the connection of the network
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, nodes_hl1])),
                     'biases':tf.Variable(tf.random_normal(nodes_hl1))}
    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
                     'biases': tf.Variable(tf.random_normal(nodes_hl1))}
    hidden_layer3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
                     'biases': tf.Variable(tf.random_normal(nodes_hl1))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl3, n_classes])),
                     'biases': tf.Variable(tf.random_normal(nodes_hl1))}

    #update weight
    l1 = tf.add(tf.matmul(data, hidden_layer1['weights']) + hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer2['weights']) + hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer3['weights']) + hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch_loss = 0
        for epoch in range(epochs):
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)


        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
'''


def train(train_images, train_labels, init_model, epoch):
    if init_model != None:
        model = init_model
    else:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(120, 120)),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=epoch)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")

def load(filename):
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    return loaded_model

def processData(dirname, filename):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + dirname + '/' + filename
    print("reading...")
    with open(dirname + '/' + filename) as f:
        data = json.load(f)

    label = []
    img = []
    class_names = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    for d in data:

        img.append(data[d]['img'])
        index = int(math.ceil(data[d]['age'] / 10)) - 1
        label.append(index)
        print(data[d]['age'])
        print(index)

    return img, label
    #print("img",img)
    #print("data",data)


def main():
    img,label = processData("Json", "4.json")
    model = load("model.json")
    train(np.array(img), np.array(label), model, len(label))

main()