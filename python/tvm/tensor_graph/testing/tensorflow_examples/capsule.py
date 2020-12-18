import numpy as np
import tensorflow as tf
import time

class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # PrimaryCaps layer
                # input: [batch_size, 20, 20, 256]
                assert input.get_shape() == [batch_size, 20, 20, 256]

                capsules = []
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID", activation_fn=None)
                        caps_i = tf.reshape(caps_i, shape=(batch_size, -1, 1, 1))
                        capsules.append(caps_i)
                assert capsules[0].get_shape() == [batch_size, 1152, 1, 1]
                capsules = tf.concat(capsules, axis=2)
                
                capsules = tf.reshape(capsules, (batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                assert capsules.get_shape() == [batch_size, 1152, 8, 1]
                return (capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(np.zeros([batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return (capsules)


def routing(input, b_IJ):
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32)

    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    # u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    u_hat_stopped = u_hat

    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            if r_iter == 3 - 1:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)

                v_J = squash(s_J)
                assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]
            elif r_iter < 3 - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]              
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]
                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # self.X = tf.constant(np.random.rand(batch_size, 28, 28, 1).astype("float32"))
                # self.labels = tf.constant(np.random.randint(0, 9, [batch_size]))
                self.X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(batch_size, ))
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.margin_loss)
            else:
                self.X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(batch_size, ))
                self.Y_ = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                self.Y = tf.reshape(self.Y_, shape=(batch_size, 10, 1))
                # self.Y = tf.reshape(self.labels, shape=(batch_size, 10, 1))
                self.build_arch()

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            # default: activation_fn=nn.relu
            # So ReLU operator is automatically included in conv1
            assert conv1.get_shape() == [batch_size, 20, 20, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape() == [batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1) # Inference output: caps2
            # v_length for loss computation
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True))


    def loss(self):
        max_l = tf.sqrt(tf.maximum(0., 0.9 - self.v_length))
        max_r = tf.sqrt(tf.maximum(0., self.v_length - 0.1))
        assert max_l.get_shape() == [batch_size, 10, 1, 1]
        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))

        T_c = self.Y
        L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

if __name__ == "__main__":
    batch_size = 32
    training = True

    model = CapsNet(is_training=training)

    supervisor = tf.train.Supervisor(graph=model.graph)

    # config=tf.ConfigProto(allow_soft_placement=True)
    with tf.device('GPU:'+str(0)):
        with supervisor.managed_session() as sess:
            number = 10
            repeats = 10

            X_np = np.random.randn(batch_size, 28, 28, 1)
            Y_np = np.random.randint(0, 9, size=batch_size)
            if training == True:
                loss_val, _ = sess.run([model.margin_loss, model.train_op], feed_dict=
                                    {model.X: X_np, model.labels: Y_np})
                
                for i in range(number):
                    records = []
                    for j in range(repeats):
                        start_time = time.time()
                        loss_val, _, = sess.run([model.margin_loss, model.train_op], feed_dict=
                                    {model.X: X_np, model.labels: Y_np})
                        end_time = time.time()
                        records.append(end_time - start_time)
                    print("Average training latency {} ms".format(1000. * np.mean(records)))
                    print("Median training latency {} ms".format(1000. * np.median(records)))
            else:
                caps2, = sess.run([model.caps2], feed_dict=
                                {model.X: X_np, model.labels: Y_np})
                for i in range(number):
                    records = []
                    for j in range(repeats):
                        start_time = time.time()
                        caps2, = sess.run([model.caps2], feed_dict=
                                {model.X: X_np, model.labels: Y_np})
                        end_time = time.time()
                        records.append(end_time - start_time)
                    print("Average inference latency {} ms".format(1000. * np.mean(records)))
                    print("Median inference latency {} ms".format(1000. * np.median(records)))
            print("batch = ", batch_size)
