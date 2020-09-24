# -------------------------------------------------------------------------------------------------------------
# 层数的变化
# -------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import neighbors
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix

# -------------------------------------------------------------------------------------------------------------

TRAINING_DATASET = "./data/out.csv"
df = pd.read_csv(TRAINING_DATASET, index_col=False, error_bad_lines=False)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# -------------------------------------------------------------------------------------------------------------

_, n_input = X_train.shape
X = tf.placeholder("float", [None, n_input])

# -------------------------------------------------------------------------------------------------------------

n_hidden_1 = 40
n_hidden_2 = 20
n_hidden_3 = 10
n_hidden_4 = 2

# -------------------------------------------------------------------------------------------------------------

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),

    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),

    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], )),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),

    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


# -------------------------------------------------------------------------------------------------------------

def encoder(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    return layer_4


# -------------------------------------------------------------------------------------------------------------

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# -------------------------------------------------------------------------------------------------------------
# training
# -------------------------------------------------------------------------------------------------------------

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(20001):
            _, c = sess.run([optimizer, cost], feed_dict={X: X_train})

            # ----------------------------------------------------------------------------------------------------------------------
            # Show loss function
            # ----------------------------------------------------------------------------------------------------------------------

            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            # ----------------------------------------------------------------------------------------------------------------------
            # Drawing
            # ----------------------------------------------------------------------------------------------------------------------

            if epoch % 100 == 0:
                encoder_result = sess.run(encoder_op, feed_dict={X: X_train})
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=y_train, cmap="rainbow", s=1)
                plt.setp(ax, xticks=[], yticks=[])
                plt.savefig('./images/' + str(epoch) + '.png')
                plt.savefig('./out_auto_encoder_4.jpg')
            # ----------------------------------------------------------------------------------------------------------------------
            # save
            # ----------------------------------------------------------------------------------------------------------------------

            if epoch % 10000 == 0:
                saver.save(sess, './model/model.ckpt')


# -------------------------------------------------------------------------------------------------------------
# test
# -------------------------------------------------------------------------------------------------------------

def test():

    # ----------------------------------------------------------------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------------------------------------------------------------

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model/')
    if ckpt:
        saver.restore(sess, ckpt)

    # ----------------------------------------------------------------------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------------------------------------------------------------------

    decoder_result = sess.run(decoder_op, feed_dict={X: X_test})
    score = clf.score(decoder_result, y_test)

    y_prediction = clf.predict(decoder_result)
    class_names = ['1', '2', '3', '4', '5']

    plot_confusion_matrix(clf, decoder_result, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='pred')
    plt.savefig('./out_auto_encoder.jpg')
    plt.show()
    out = pd.crosstab(y_test, y_prediction, rownames=['True'], colnames=['Predicted'], margins=True)

    print('accuracy of test: ', score)
    print('confusion_matrix: \n', out)




# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
     train()
    #test()

# ----------------------------------------------------------------------------------------------------------------------
