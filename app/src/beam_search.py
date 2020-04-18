"""This file contains code to run beam search decoding"""

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import data


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""
    def __init__(self, tokens, log_probs, state, attn_dists, coverage):
        """Hypothesis constructor.

        Args:
            tokens: List of integers. The ids of the tokens that form the summary so far.
            log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
            state: Current state of the decoder, a LSTMStateTuple.
            attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
            p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
            coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, coverage):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
            token: Integer. Latest token produced by beam search.
            log_prob: Float. Log prob of the latest token.
            state: Current decoder state, a LSTMStateTuple.
            attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen: Generation probability on latest step. Float.
            coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
            New Hypothesis for next step.
        """
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
            coverage=coverage
            )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)


class BeamSearch(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 dec_sequence_length, enc_sequence_length,
                 rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std,
                 use_coverage=True, beam_size=4, min_dec_steps=3,
                 ):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dec_sequence_length = dec_sequence_length
        self.enc_sequence_length = enc_sequence_length
        self.random_norm_init = tf.random_normal_initializer(stddev=rand_norm_init_std)
        self.rand_unif_init = tf.random_uniform_initializer(-rand_unif_init_mag, rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=trunc_norm_init_std)
        self.use_coverage = use_coverage
        self.beam_size = beam_size
        self.min_dec_steps = min_dec_steps

    def add_placeholders(self):
        self.enc_padding_mask = tf.placeholder(tf.float32, [None, None], name='enc_padding_mask')
        self.dec_seq_batch = tf.placeholder(tf.int32, shape=[None, 1], name='dec_batch')
        self.encoder_states = tf.placeholder(tf.float32, shape=[None, None, 2 * self.hidden_dim], name='encoder_states')
        self.fw_st = tf.placeholder(tf.float32, shape=[None, self.hidden_dim], name='encoder_fw_states')
        self.fw_op = tf.placeholder(tf.float32, shape=[None, self.hidden_dim], name='encoder_fw_output')
        self.prev_coverage = tf.placeholder(tf.float32, [None, None], name='prev_coverage')

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
            context_vector = tf.reduce_sum(tf.reshape(attn_dist, [tf.shape(self.encoder_states)[0], -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
            context_vector = tf.reshape(context_vector, [-1, attn_size])

        return context_vector, attn_dist, coverage

    def decoder_unit(self, inp, prev_state):
        cell_gen = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        return cell_gen(inp, prev_state)
        # return tf.nn.xw_plus_b(cell_output, dout_w, dout_b), state

    def add_decoder(self):
        with tf.variable_scope('searcher'):
            with tf.variable_scope('embedding') as demb_scope:
                d_embedding = tf.get_variable('embedding', [self.num_emb, self.emb_dim], dtype=tf.float32,
                                              initializer=self.trunc_norm_init)
                emb_dec_inputs = tf.transpose(tf.nn.embedding_lookup(d_embedding, self.dec_seq_batch),
                                              perm=[1, 0, 2])
            with tf.variable_scope('decoding') as dcode_scope:
                lst_context_v = []
                lst_cell_out = []
                self.attn_dists = []
                trg_unstackd = tf.unstack(emb_dec_inputs, axis=0)
                self.state = LSTMStateTuple(self.fw_op, self.fw_st)
                # for attention
                # attn_size = self.encoder_states.get_shape()[2].value
                # context_vector = tf.zeros([tf.shape(self.encoder_states)[0], attn_size])
                prev_coverage = self.prev_coverage if self.use_coverage else None
                if prev_coverage is not None:  # for beam search mode with coverage
                    # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
                    prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)
                context_vector, _, coverage = self.attention(self.state, prev_coverage)
                for i, inp in enumerate(trg_unstackd):
                    inp = self.linear([inp] + [context_vector], self.emb_dim, True)
                    cell_output, self.state = self.decoder_unit(inp, self.state)
                    dcode_scope.reuse_variables()
                    context_vector, attn_dist, _ = self.attention(self.state, coverage)
                    lst_cell_out.append(cell_output)
                    lst_context_v.append(context_vector)
                    self.attn_dists.append(attn_dist)
            with tf.variable_scope("output_projection") as out_project_scope:
                dout_w = tf.get_variable('w_out', [self.hidden_dim, self.num_emb],
                                         dtype=tf.float32, initializer=self.rand_unif_init)
                dout_b = tf.get_variable('b_out', [self.num_emb],
                                         dtype=tf.float32, initializer=self.rand_unif_init)
                g_predictions = []
                for i, (context_vector, cell_output) in enumerate(zip(lst_context_v, lst_cell_out)):
                    cell_output = self.linear([cell_output, context_vector], self.hidden_dim, True)
                    logits = tf.nn.xw_plus_b(cell_output, dout_w, dout_b)
                    g_predictions.append(tf.nn.softmax(logits))

            # batch_size = self.encoder_states.get_shape()[0].value
            final_dists = g_predictions[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, self.beam_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)
            if coverage is not None:
                batch_size = tf.shape(self.encoder_states)[0]  # encoder_states.get_shape()[0].value
                self.coverage = tf.reshape(coverage, [batch_size, -1])

    def decode_onestep(self, sess, feed):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        to_return = {
            "ids": self._topk_ids, "probs": self._topk_log_probs, "states": self.state, "attn_dists": self.attn_dists}
        if self.use_coverage:
            to_return['coverage'] = self.coverage
        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(self.beam_size)]
        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if self.use_coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == self.beam_size
        else:
            new_coverage = [None for _ in range(self.beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, new_coverage

    def run_beam_search(self, sess, generator, vocab, batch):
        """Performs beam search decoding on the given example.

        Args:
            sess: a tf.Session
            model: a seq2seq model
            vocab: Vocabulary object
            batch: Batch object that is the same example repeated across the batch
        Returns:
            best_hyp: Hypothesis object; the best hypothesis found by beam search.
        """
        enc_st, dec_in = sess.run([generator.encoder_states, generator._dec_in_state],
                                  feed_dict={generator.src_seq_batch: np.tile(batch.enc_batch, (self.beam_size, 1)),
                                             generator.enc_padding_mask: np.tile(batch.enc_padding_mask, (self.beam_size, 1)),
                                             #generator.seq_len_batch: np.tile(batch.enc_lens, (self.beam_size, 1))
                                             })
        dec_in_tpl = tf.contrib.rnn.LSTMStateTuple(dec_in.c[0], dec_in.h[0])
        # Initialize beam_size-many hypotheses
        hyps = [
            Hypothesis(
                tokens=[vocab.word2id(data.START_DECODING)],
                log_probs=[0.0],
                state=dec_in_tpl,
                attn_dists=[],
                coverage=np.zeros([batch.enc_batch.shape[1]])  # zero vector of length attention_length
            )
            for _ in range(self.beam_size)]
        results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)

        steps = 0
        while steps < self.dec_sequence_length and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
            latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
            states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
            prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

            feed = self.feed_dict(batch, enc_st, states, latest_tokens)

            if self.use_coverage:
                feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)

            # Run one step of the decoder to get the new info
            topk_ids, topk_log_probs, new_states, attn_dists, new_coverage = self.decode_onestep(sess, feed)

            # Extend each hypothesis and collect them all in all_hyps
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
            for i in range(num_orig_hyps):
                h, new_state, attn_dist, new_coverage_i = hyps[i], new_states[i], attn_dists[i], new_coverage[i]# take the ith hypothesis and new decoder state info
                for j in range(self.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    # Extend the ith hypothesis with the jth option
                    new_hyp = h.extend(topk_ids[i, j], topk_log_probs[i, j], state=new_state, attn_dist=attn_dist, coverage=new_coverage_i)
                    all_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            hyps = [] # will contain hypotheses for the next step
            for h in self.sort_hyps(all_hyps): # in order of most likely h
                if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= self.min_dec_steps:
                        results.append(h)
                else: # hasn't reached stop token, so continue to extend this hypothesis
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(results) == self.beam_size:
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break
            steps += 1

        # At this point, either we've got beam_size results, or we've reached maximum decoder steps
        if len(results) == 0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
            results = hyps

        # Sort hypotheses by average log probability
        hyps_sorted = self.sort_hyps(results)

        # Return the hypothesis with highest average log prob
        return hyps_sorted[0], dec_in

    def sort_hyps(self, hyps):
        """Return a list of Hypothesis objects, sorted by descending average log probability"""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

    def feed_dict(self, batch, encoder_states, dec_init_states, latest_tokens):
        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        # new_dec_in_state = LSTMStateTuple(new_c, new_h)
        return {
            self.enc_padding_mask: batch.enc_padding_mask,
            self.dec_seq_batch: np.transpose(np.array([latest_tokens])),
            self.encoder_states: encoder_states,
            self.fw_op: new_c,
            self.fw_st: new_h,
        }

    def get_search_replc_ops(self):
        dict_vname = {}
        gvars_placeholder = []
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
            ph = tf.placeholder(v.dtype, v.get_shape())
            gvars_placeholder += [ph]
            dict_vname.update({v.name: ph})
        replace_ops = {}
        for i, v in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='searcher')):
            replace_ops.update({v.name.replace('searcher','generator'): tf.assign(v, dict_vname[v.name.replace('searcher','generator')])})
        return gvars_placeholder, replace_ops

    def replace_search_w_generator(self, sess, gen_placeholder, beam_replace_ops):
        for i, v in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')):
            sess.run(beam_replace_ops[v.name], feed_dict={gen_placeholder[i]: sess.run(v)})