"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import shutil
import pyrouge
import logging
import data
import numpy as np


def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config


def restore_best_model(log_dir, eval_type='eval'):
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=get_config())
    print("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.global_variables() if "Adam" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = load_ckpt(saver, sess, log_dir, eval_type)
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(log_dir, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def load_ckpt(saver, sess, log_dir, ckpt_dir="train"):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      latest_filename = "checkpoint_best" if ckpt_dir=="eval" or 'eval_disc' else None
      ckpt_path = os.path.join(log_dir, ckpt_dir)
      tf.logging.info('checkpoint path %s', ckpt_path)
      ckpt_state = tf.train.get_checkpoint_state(ckpt_path, latest_filename=latest_filename)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    tf.logging.info(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    tf.logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)

    log_out = {} # "ROUGE-%s: " % x
    for x in ["1", "2", "l"]:
        for y in ["recall", "precision", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            val = results_dict[key]
            log_out.update({key: val})
            # log_out += "%s: %.4f " % (key, val)
    return log_out


def write_for_rouge(reference_sents, decoded_sents, ex_index, _rouge_ref_dir, _rouge_dec_dir):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx,sent in enumerate(reference_sents):
            f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
        for idx,sent in enumerate(decoded_sents):
            f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    # tf.logging.info("Wrote example %i to file" % ex_index)


def rouge_eval_bsearch(sess, generator, searcher, data_loader, vocab, rouge_ref_dir, rouge_dec_dir, log_dir, hps=None,
                       extra_info=False):
    if os.path.exists(rouge_ref_dir):
        shutil.rmtree(rouge_ref_dir)
    if os.path.exists(rouge_dec_dir):
        shutil.rmtree(rouge_dec_dir)
    os.mkdir(rouge_ref_dir)
    os.mkdir(rouge_dec_dir)
    counter = 0
    gseq_lst = []
    gdin_lst = []
    article_lst = []
    decoded_output_list = []
    original_sent_list = []
    while True:
        batch = data_loader.next_batch()
        if batch is None:
            results_dict = rouge_eval(rouge_ref_dir, rouge_dec_dir)
            rouge_out = rouge_log(results_dict, log_dir)
            break
        best_hyp, din = searcher.run_beam_search(sess, generator, vocab, batch)
        # Extract the output ids from the hypothesis and convert back to words
        output_ids = np.array(best_hyp.tokens[1:], dtype='int32')
        # Extract the output ids from the hypothesis and convert back to words
        decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if hps.pointer_gen else None))
        # decoded_words = data.outputids2words(output_ids, vocab, None)
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words)  # single string
        original_sent = batch.original_abstracts[0]
        original_artc = batch.original_articles[0]
        write_for_rouge([original_sent], [decoded_output], counter, rouge_ref_dir, rouge_dec_dir)
        print '%s\n%s\n%s\n\n'%(original_artc, original_sent, decoded_output)
        #
        # for original_art, original_sent, output_ids in equizip(batch.enc_batch, batch.original_abstracts, generated_samples):
        #     decoded_words = data.outputids2words(output_ids, vocab, None)
        #     # Remove the [STOP] token from decoded_words, if necessary
        #     try:
        #         fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
        #         decoded_words = decoded_words[:fst_stop_idx]
        #     except ValueError:
        #         decoded_words = decoded_words
        #     decoded_output = ' '.join(decoded_words)  # single string
        #     write_for_rouge([original_sent], [decoded_output], counter, _rouge_ref_dir, _rouge_dec_dir)  # write ref and decoded to eval them later
        counter += 1  # this is how many examples we've decoded
        try:
            output_tmp = np.array([vocab.word2id(data.STOP_DECODING)] * batch.target_batch.shape[1])
            output_tmp[:len(output_ids)] = output_ids
        except Exception as e:
            print e.args
            print batch.target_batch.shape[1]
            print output_ids
            raise
        gseq_lst.append(output_tmp)
        gdin_lst.append(din)
        article_lst.append(original_artc)
        decoded_output_list.append(decoded_output)
        original_sent_list.append(original_sent)
    if extra_info:
        return rouge_out, gseq_lst, gdin_lst, article_lst, original_sent_list, decoded_output_list
    else:
        return rouge_out, gseq_lst, gdin_lst, article_lst


def rouge_evaluation(sess, generator, data_loader, vocab, _rouge_ref_dir, _rouge_dec_dir, log_dir, rollout=None):
    if os.path.exists(_rouge_ref_dir):
        shutil.rmtree(_rouge_ref_dir)
    if os.path.exists(_rouge_dec_dir):
        shutil.rmtree(_rouge_dec_dir)
    os.mkdir(_rouge_ref_dir)
    os.mkdir(_rouge_dec_dir)
    sample_org_abs_sents = []
    sample_org_art_sents = []
    sample_dec_abs_sents = []
    counter = 0
    while True:
        batch = data_loader.next_batch()
        if batch is None:
            results_dict = rouge_eval(_rouge_ref_dir, _rouge_dec_dir)
            rouge_out = rouge_log(results_dict, log_dir)
            break
        if rollout is not None:
            samples_, enc_st, dec_in = generator.generate(
                sess, batch, _returns=[generator.gen_x, generator.encoder_states, generator._dec_in_state])
            g_num = np.random.choice(np.arange(1, 3), size=1)[-1]
            feed = {rollout.src_seq_batch: batch.enc_batch,
                    rollout.seq_len_batch: batch.enc_lens,
                    rollout.enc_padding_mask: batch.enc_padding_mask,
                    rollout.trg_seq_batch: samples_,
                    # rollout.encoder_states: enc_st,
                    # rollout.fw_op: dec_in[0],
                    # rollout.fw_st: dec_in[1],
                    rollout.given_num: g_num}
            generated_samples = rollout.get_sample(sess, feed)
        else:
            generated_samples = generator.generate(sess, batch)

        for original_art, original_sent, output_ids in zip(batch.enc_batch, batch.original_abstracts, generated_samples):
            decoded_words = data.outputids2words(output_ids, vocab, None)
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words)  # single string
            write_for_rouge([original_sent], [decoded_output], counter, _rouge_ref_dir, _rouge_dec_dir)  # write ref and decoded to eval them later
            counter += 1  # this is how many examples we've decoded
            if np.random.random() < 0.005:
                art_words = data.outputids2words(original_art, vocab, None)
                sample_org_art_sents.append(' '.join(art_words))
                sample_org_abs_sents.append(original_sent)
                sample_dec_abs_sents.append(decoded_output)
    for i, (art, org, dec) in enumerate(zip(sample_org_art_sents, sample_org_abs_sents, sample_dec_abs_sents)):
        if i < 2:
            print art
            print org
            print dec
            print '\n'
    return rouge_out


def reformat_samples(samples, eos, pad):
    rows, stop_indices = np.where(samples == eos)
    p_batch_pad = np.ones_like(samples, dtype='int32') * pad
    p_batch_mask = np.ones_like(samples, dtype='int32')
    for r_inx, stop_pos in zip(rows, stop_indices):
        p_batch_mask[r_inx, stop_pos + 1:] = 0
    return samples * p_batch_mask + p_batch_pad * (1 - p_batch_mask)


def create_discriminator_data(t_batch, negative_examples):
    # negative_examples = reformat_samples(negative_examples , eos, pad)
    sentences = np.concatenate([t_batch, negative_examples], 0)
    # Generate labels
    positive_labels = [[0, 1] for _ in t_batch]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return sentences, labels