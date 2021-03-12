import tensorflow as tf 
g=tf.Graph()
with g.as_default():
    v_1 = tf.constant([1, 2, 3, 4])
    v_2 = tf.constant([2, 3, 4, 5])
    v_add = tf.add(v_1, v_2)
sess=tf.compat.v1.Session(graph=g)
print(sess.run(v_add))