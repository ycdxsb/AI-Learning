import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def main():
    batchsz = 32
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(batchsz)

    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(10)
    ])
    model.build(input_shape=[None, 28*28])
    model.summary()

    optimizer = optimizers.Adam(lr=1e-3)
    for epoch in range(20):
        for step, (x, y) in enumerate(db_train):
            # x:[b,28,28]
            # y:[b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b,784] => [b,10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(
                    y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))
        
        total_correct = 0
        total_num = 0
        for x,y in db_test:
            x = tf.reshape(x,[-1,28*28])
            logits=model(x)
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            
            correct = tf.equal(pred,y)
            correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))
            
            total_correct+=int(correct)
            total_num+=x.shape[0]
        print(epoch,'test acc:',"%d/%d"%(total_correct,total_num))


if __name__ == "__main__":
    main()
