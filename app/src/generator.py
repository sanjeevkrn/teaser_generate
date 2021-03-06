import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 dec_sequence_length, start_token,
                 enc_sequence_length,
                 rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std, adagrad_init_acc=0.1,
                 learning_rate=0.01,
                 grad_clip=5.0,
                 use_coverage=False,
                 use_coverage_loss=False
                 ):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dec_sequence_length = dec_sequence_length
        self.enc_sequence_length = enc_sequence_length
        self.start_token = start_token
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.adagrad_init_acc = adagrad_init_acc
        self.grad_clip = grad_clip

        self.rand_unif_init = tf.random_uniform_initializer(-rand_unif_init_mag, rand_unif_init_mag)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=trunc_norm_init_std)
        self.rand_norm_init = tf.random_normal_initializer(stddev=rand_norm_init_std)

        self.use_coverage = use_coverage
        self.use_coverage_loss = use_coverage_loss

    def add_placeholders(self):
        self.src_seq_batch = tf.placeholder(tf.int32, shape=[None, None], name='enc_batch') # sequence of tokens given to encoder
        self.enc_padding_mask = tf.placeholder(tf.float32, [None, None], name='enc_padding_mask')
        self.seq_len_batch = tf.placeholder(tf.int32, shape=[None], name='enc_lens')
        self.trg_seq_batch = tf.placeholder(tf.int32, shape=[None, self.dec_sequence_length], name='target_batch') # sequence of tokens generated by generator
        self.dec_seq_batch = tf.placeholder(tf.int32, shape=[None, self.dec_sequence_length], name='dec_batch') # sequence of tokens generated by generator
        self.dec_padding_mask = tf.placeholder(tf.float32, [None, self.dec_sequence_length], name='dec_padding_mask')
        self.start_dec = tf.placeholder(tf.int32, shape=[None], name='dec_start')
        self.g_smax = tf.placeholder(tf.float32, shape=[None, self.dec_sequence_length, self.num_emb], name='gsmax')
        self.given_num = tf.placeholder(tf.int32)

    def add_encoder(self):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder') as ecode_scope:
            # Add embedding matrix
            with tf.variable_scope('embedding') as eemb_scope:
                embedding = tf.get_variable('embedding', [self.num_emb, self.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

                emb_enc_inputs = tf.transpose(tf.nn.embedding_lookup(embedding, self.src_seq_batch), perm=[1, 0, 2])
            # Add the encoder.
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init) #initializer=self.rand_unif_init)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init) #initializer=self.rand_unif_init))
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_enc_inputs,
                                                                                dtype=tf.float32,
                                                                                # sequence_length=self.seq_len_batch,
                                                                                # swap_memory=True,)
                                                                                time_major=True)
            encoder_outputs = tf.transpose(tf.concat(axis=2, values=encoder_outputs), perm=[1, 0, 2]) # concatenate the forwards and backwards states
            self.encoder_states = encoder_outputs
            # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
            self._dec_in_state = bw_st # self._reduce_states(fw_st, bw_st)

    def linear(self, args, output_size, bias, bias_start=0.0, scope=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_start: starting value to initialize the bias; 0 by default.
          scope: VariableScope for the created subgraph; defaults to "Linear".

        Returns:
          A 2D Tensor with shape [batch x output_size] equal to
          sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (isinstance(args, (list, tuple)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(axis=1, values=args), matrix)
            if not bias:
                return res
            bias_term = tf.get_variable(
                "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term

    def attention(self, decoder_state, coverage=None):
        """Calculate the context vector and attention distribution from the decoder state.

        Args:
            decoder_state: state of the decoder
            coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

        Returns:
            context_vector: weighted sum of encoder_states
            attn_dist: attention distribution
            coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
        """
        with tf.variable_scope("attention_compute"):
            attn_size = 2*self.hidden_dim
            batch_size = tf.shape(self.encoder_states)[0]
            # Reshape encoder_states (need to insert a dim)
            encoder_states = tf.expand_dims(self.encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)
            # To calculate attention, we calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            # where h_i is an encoder state, and s_t a decoder state.
            # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
            # We set it to be equal to the size of the encoder states.
            attention_vec_size = attn_size

            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

            # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
            decoder_features = self.linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

            w_c = tf.get_variable("w_cvg", [1, 1, 1, attention_vec_size])
            if self.use_coverage and coverage is not None:  # non-first step of coverage
                # Multiply coverage vector by w_c to get coverage_features.
                coverage_features = tf.nn.conv2d(coverage, w_c, [1, 1, 1, 1],
                                                 "SAME")  # c has shape (batch_size, attn_length, 1, attention_vec_size)

                # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                e = tf.reduce_sum(tf.tanh(encoder_features + decoder_features + coverage_features), [2, 3]) # calculate e

                # Calculate attention distribution
                attn_dist = tf.nn.softmax(e * self.enc_padding_mask) # masked_attention(e)
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                attn_dist /= tf.reshape(masked_sums, [-1, 1])  # re-normalize
                # Update coverage vector
                coverage += tf.reshape(attn_dist, [tf.shape(self.encoder_states)[0], -1, 1, 1])
            else:
                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = tf.reduce_sum(tf.tanh(encoder_features + decoder_features), [2, 3])  # calculate e
                # Calculate attention distribution
                attn_dist = tf.nn.softmax(e * self.enc_padding_mask)  # masked_attention(e)
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                attn_dist /= tf.reshape(masked_sums, [-1, 1])  # re-normalize
                if self.use_coverage:  # first step of training
                    coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage
            # Calculate the context vector from attn_dist and encoder_states
            context_vector = tf.reduce_sum(tf.reshape(attn_dist, [tf.shape(self.src_seq_batch)[0], -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
            context_vector = tf.reshape(context_vector, [-1, attn_size])

        return context_vector, attn_dist, coverage

    def decoder_unit(self, inp, prev_state):
        cell_gen = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True, initializer=tf.random_normal_initializer(stddev=0.1), reuse=tf.AUTO_REUSE)#initializer=self.rand_unif_init)
        return cell_gen(inp, prev_state)
        # return tf.nn.xw_plus_b(cell_output, dout_w, dout_b), state

    def add_generator(self):
        with tf.variable_scope('generator'):
            with tf.variable_scope('embedding') as demb_scope:
                d_embedding = tf.get_variable('embedding', [self.num_emb, self.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)
                emb_dec_inputs = tf.transpose(tf.nn.embedding_lookup(d_embedding, self.dec_seq_batch), perm=[1, 0, 2])

            with tf.variable_scope('decoding') as dcode_scope:
                self.attn_dists = []
                lst_context_v = []
                lst_cell_out = []
                trg_unstackd = tf.unstack(emb_dec_inputs, axis=0)
                state = self._dec_in_state
                # for attention
                # batch_size = self.encoder_states.get_shape()[0].value
                attn_size = self.encoder_states.get_shape()[2].value
                context_vector = tf.zeros([tf.shape(self.src_seq_batch)[0], attn_size])
                coverage = None
                for i, inp in enumerate(trg_unstackd):
                    if i > 0:
                        dcode_scope.reuse_variables()
                    inp = self.linear([inp] + [context_vector], self.emb_dim, True)
                    cell_output, state = self.decoder_unit(inp, state)
                    context_vector, attn_dist, coverage = self.attention(state, coverage)
                    lst_context_v.append(context_vector)
                    lst_cell_out.append(cell_output)
                    self.attn_dists.append(attn_dist)
                # self.pretrain_loss_v1 = \
                #     -tf.reduce_sum(# tf.reshape(tf.reduce_sum(
                #     tf.one_hot(tf.to_int32(tf.reshape(self.trg_seq_batch, [-1])), self.num_emb, 1.0, 0.0) * \
                #     tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0))
                #     #, axis=-1), (tf.shape(self.src_seq_batch)[0], -1)
                # )
                #)

                # self.train_loss = tf.contrib.legacy_seq2seq.sequence_loss(
                #     g_logits, tf.unstack(tf.transpose(self.trg_seq_batch)),
                #     tf.unstack(tf.transpose(self.dec_padding_mask)))
            with tf.variable_scope("output_projection") as out_project_scope:
                g_predictions = []
                g_logits = []
                dout_w = tf.get_variable('w_out', [self.hidden_dim, self.num_emb],
                                         dtype=tf.float32, initializer=self.rand_unif_init)
                dout_b = tf.get_variable('b_out', [self.num_emb],
                                         dtype=tf.float32, initializer=self.rand_unif_init)
                for i, (context_vector, cell_output) in enumerate(zip(lst_context_v, lst_cell_out)):
                    if i > 0:
                        out_project_scope.reuse_variables()
                    cell_output = self.linear([cell_output, context_vector], self.hidden_dim, True)
                    logits = tf.nn.xw_plus_b(cell_output, dout_w, dout_b)
                    g_logits.append(logits)
                    g_predictions.append(tf.nn.softmax(logits))

            self.g_predictions = tf.transpose(tf.stack(g_predictions, axis=0), perm=[1, 0, 2])
            log_perp_list = []
            for logit, target, weight in zip(g_logits, tf.unstack(tf.transpose(self.trg_seq_batch)),
                                             tf.unstack(tf.transpose(self.dec_padding_mask))):
                target = tf.reshape(target, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit)
                log_perp_list.append(crossent * weight)
            log_perps = tf.add_n(log_perp_list)
            log_perps /= tf.reduce_sum(self.dec_padding_mask, axis=1)
            self.train_loss = tf.reduce_sum(log_perps)

            if self.use_coverage and self.use_coverage_loss:
                coverage = tf.zeros_like(self.attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
                covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
                for a in self.attn_dists:
                    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
                    covlosses.append(covloss)
                    coverage += a  # update the coverage vector

                dec_lens = tf.reduce_sum(self.dec_padding_mask, axis=1)  # shape batch_size. float32
                values_per_step = [v * self.dec_padding_mask[:, dec_step] for dec_step, v in enumerate(covlosses)]
                values_per_ex = tf.add_n(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
                coverage_loss = tf.reduce_sum(values_per_ex)  # overall average
                tf.summary.scalar('coverage_loss', coverage_loss)
                self.train_loss = self.train_loss + coverage_loss

            tf.summary.scalar('genr_loss', self.train_loss, collections=['genr_summaries'])

            ################################################################################
            gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.dec_sequence_length,
                                                 dynamic_size=False, infer_shape=True)
            gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.dec_sequence_length,
                                                 dynamic_size=False, infer_shape=True)

            def _g_recurrence(i, x_t, h_tm1, gen_x, c_vector):
                with tf.variable_scope('embedding', reuse=True) as demb_scope:
                    xt_emb = tf.nn.embedding_lookup(d_embedding, x_t)
                with tf.variable_scope('decoding', reuse=True) as dcode_scope:
                    inp = self.linear([xt_emb, c_vector], self.emb_dim, True)
                    cell_output, h_t = self.decoder_unit(inp, h_tm1)
                    c_vector, attn_dist, _ = self.attention(h_t, None)
                with tf.variable_scope("output_projection", reuse=True) as out_project_scope:
                    cell_output = self.linear([cell_output, c_vector], self.hidden_dim, True)
                    logits = tf.nn.xw_plus_b(cell_output, dout_w, dout_b)
                # next_token = tf.cast(tf.argmax(tf.nn.softmax(o_t), -1), tf.int32)
                log_prob = tf.log(tf.nn.softmax(logits))
                next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [tf.shape(self.src_seq_batch)[0]]),
                                     tf.int32)
                # gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0), tf.nn.softmax(logits)), -1))
                # gen_o = gen_o.write(i, tf.nn.softmax(logits))
                gen_x = gen_x.write(i, next_token)  # indices, batch_size
                return i + 1, next_token, h_t, gen_x, c_vector

            _, _, _, self.gen_x, _ = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self.dec_sequence_length,
                body=_g_recurrence,
                loop_vars=(tf.constant(0, dtype=tf.int32),
                           tf.fill(tf.shape(self.seq_len_batch), self.start_token),
                           self._dec_in_state,
                           gen_x,
                           tf.zeros([tf.shape(self.src_seq_batch)[0], attn_size]),
                           #tf.zeros_like(self.attn_dists)
                           ))

            self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1, 0])  # batch_size x seq_length
            # self.gen_o = tf.transpose(self.gen_o.stack(), perm=[1, 0, 2])  # batch_size x seq_length

            # self.entropy = C_E * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)

            # self.entropy = -tf.reduce_sum(tf.reduce_sum(
            #         tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 0.99)
            #         * tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 0.99)),
            #     axis=1) * tf.reshape(self.dec_padding_mask, [-1])
            # )#/tf.reduce_sum(self.dec_padding_mask)

            # tf.summary.scalar('advr_genr_entropy', self.entropy, collections=['advr_summaries'])

    def generate(self, sess, batch, _returns=None):
        feed = self._make_feed_dict(batch)
        if _returns is not None:
            return sess.run(_returns, feed)
        return sess.run(self.gen_x, feed)

    def _make_feed_dict(self, batch):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self.src_seq_batch] = batch.enc_batch
        feed_dict[self.enc_padding_mask] = batch.enc_padding_mask
        feed_dict[self.seq_len_batch] = batch.enc_lens
        # feed_dict[self.dec_padding_mask] = batch.dec_padding_mask
        feed_dict[self.dec_seq_batch] = batch.dec_batch
        feed_dict[self.trg_seq_batch] = batch.target_batch
        feed_dict[self.dec_padding_mask] = np.ones_like(batch.target_batch)
        return feed_dict

    def train_step(self, sess, batch, to_return, global_step=None):
        feed = self._make_feed_dict(batch)
        outputs = sess.run([self.train_updates, self.train_loss, global_step]+to_return, feed)
        return outputs

    def add_train_opt(self, global_step=None):
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
        self.train_opt = self.g_optimizer(self.learning_rate) #, self.adagrad_init_acc)
        self.train_grad, _ = tf.clip_by_global_norm(tf.gradients(self.train_loss, g_params+e_params), self.grad_clip)
        self.train_updates = self.train_opt.apply_gradients(zip(self.train_grad, g_params+e_params), global_step=global_step)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)
        # return tf.train.AdagradOptimizer(*args, **kwargs)

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

