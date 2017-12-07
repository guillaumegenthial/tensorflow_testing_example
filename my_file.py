import tensorflow as tf


def get_entry_tf(t, indices_d1, indices_d2, batch_size):
    """
    Args:
        t: shape = [batch, d1, d2]
        indices_1d: shape = [batch]
        indices_2d: shape = [batch]

    Returns:
        o: shape = [batch], with o[i] = t[i, indices_1d[i], indices_2d[i]]

    """
    indices = tf.stack([tf.range(batch_size), indices_d1, indices_d2], axis=1)
    return tf.gather_nd(t, indices)