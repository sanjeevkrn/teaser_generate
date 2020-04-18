import sys
import argparse
import numpy as np
import tensorflow as tf
import random

import time

from sklearn.model_selection import ParameterGrid

from data import Vocab
from batcher import Batcher
from collections import namedtuple
import os
from util import rouge_eval_bsearch, get_config
from pointer_generator import GeneratorPoint
from beam_search_point import BeamSearchPoint
from generator import Generator
from beam_search import BeamSearch
import data


def setup_argparser():
    parser = argparse.ArgumentParser(prog='teaser_generation')

    parser.add_argument('-v', '--version', action='version', version='0.1')
    parser.add_argument('-e', "--experiment_dir", type=str, help="folder for data experiment logs", default=None)
    parser.add_argument('--pointer', dest='pointer_gen', action='store_true', help="Use seq2seq_pointer")
    parser.add_argument('--no_pointer', dest='pointer_gen', action='store_false', help="Use seq2seq")
    parser.set_defaults(pointer_gen=False)
    parser.add_argument('--coverage_loss', dest='coverage_loss', action='store_true', help="Use coverage loss")
    parser.add_argument('--no_coverage_loss', dest='coverage_loss', action='store_true', help="Use no coverage loss")
    parser.set_defaults(coverage=False)
    return parser


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 100000)#12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss


def build_graph(global_step, start_token, stop_token):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()

    with tf.device("/gpu:0"):
        if args.pointer_gen:
            generator = GeneratorPoint(
                VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, DEC_SEQ_LENGTH, start_token, ENC_SEQ_LENGTH,
                rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std,
                learning_rate=LEARNING_RATE, grad_clip=grad_clip)
            searcher = BeamSearchPoint(
                VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, DEC_SEQ_LENGTH, ENC_SEQ_LENGTH,
                rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std, beam_size=BEAM_SIZE)
        else:
            generator = Generator(
                VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, DEC_SEQ_LENGTH, start_token, ENC_SEQ_LENGTH,
                rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std,
                learning_rate=LEARNING_RATE, grad_clip=grad_clip,
                use_coverage_loss=args.coverage_loss
            )
            searcher = BeamSearch(
                VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, DEC_SEQ_LENGTH, ENC_SEQ_LENGTH,
                rand_unif_init_mag, trunc_norm_init_std, rand_norm_init_std, beam_size=BEAM_SIZE
            )
        generator.add_placeholders()
        generator.add_encoder()
        generator.add_generator()
        generator.add_train_opt(global_step=global_step)
        searcher.add_placeholders()
        searcher.add_decoder()

    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)
    return generator, searcher


def main():
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    vocab = Vocab(vocab_path, max_size=VOCAB_SIZE)
    hps_train_dict = {
        'batch_size': BATCH_SIZE,
        'max_dec_steps': DEC_SEQ_LENGTH,
        'max_enc_steps': ENC_SEQ_LENGTH,
        'pointer_gen':args.pointer_gen}
    hps_test_dict = {
        'batch_size': 1,
        'max_dec_steps': DEC_SEQ_LENGTH,
        'max_enc_steps': ENC_SEQ_LENGTH,
        'pointer_gen':args.pointer_gen}
    hps_train = namedtuple("HParams", hps_train_dict.keys())(**hps_train_dict)
    hps_test = namedtuple("HParams", hps_test_dict.keys())(**hps_test_dict)

    eval_dir = os.path.join(log_dir, "eval")
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')

    start_sym = vocab.word2id(data.START_DECODING)
    stop_sym = vocab.word2id(data.STOP_DECODING)
    pad_sym = vocab.word2id(data.PAD_TOKEN)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    best_score = tf.Variable(0., name='best_score', trainable=False)

    best_score_ph = tf.placeholder(best_score.dtype, best_score.get_shape())
    assign_bestscore = tf.assign(best_score, best_score_ph)

    generator, searcher,  = build_graph(global_step, start_sym, stop_sym)
    ph_gvars_sc, srch_rplc = searcher.get_search_replc_ops()

    # restore_best_model(log_dir, 'eval')

    train_dir = os.path.join(log_dir, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    summaries_genr = tf.summary.merge_all(key='genr_summaries')

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=global_step)

    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=get_config())
    tf.logging.info("Created session.")

    log = open(os.path.join(log_dir, 'experiment-log.txt'), 'a+')
    log.write(
        'hyperparams: EMB_DIM=%s, '
        'HIDDEN_DIM=%s, '
        'DEC_SEQ_LENGTH=%s, '
        'ENC_SEQ_LENGTH=%s, '
        'BATCH_SIZE=%s, '
        'LEARNING_RATE=%s, '
        'VOCAB_SIZE=%s, '
        'train_path=%s, '
        'grad_clip=%s, '
        'BEAM_SIZE=%s\n'%(
            EMB_DIM, HIDDEN_DIM, DEC_SEQ_LENGTH, ENC_SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE, VOCAB_SIZE,
            train_data_path, grad_clip, BEAM_SIZE))
    try:
        with sess_context_manager as sess:
            #  pre-train generator
            log.write('training...\n')
            tf.logging.info('running training ...')

            # # evaluate on test data
            # test_batcher = Batcher(test_data_path, vocab, hps_test, single_pass=True, num_epoch=1)
            # rouge_out, _, _, lst_article, lst_orig, lst_pred = rouge_eval_bsearch(sess, generator, searcher, test_batcher, vocab,
            #                                      rouge_ref_dir,
            #                                      rouge_dec_dir, log_dir, hps_test, extra_info=True)
            # with open(os.path.join(log_dir, 'source_orig_pred.txt'), 'wb') as art_fh:
            #     for artc, orig, pred in zip(lst_article, lst_orig, lst_pred):
            #         art_fh.write(artc+'||'+orig+'||'+pred+'\n')
            # rouge_text = 'Test: ' + ' '.join(
            #     ["%s: %.4f " % (key.replace('rouge_', ''), val) for key, val in rouge_out.items()])
            # print rouge_text
            # exit()

            best_rouge_score = sess.run(best_score)
            print '##### %.4f best saved score #####'% best_rouge_score
            running_avg_loss = 0
            for epoch in range(EPOCH_NUM_GEN):
                t0 = time.time()
                train_data_loader = Batcher(train_data_path, vocab, hps_train, single_pass=False, num_epoch=1)
                loss = 0
                while True:
                    batch = train_data_loader.next_batch()
                    if batch is None:
                        break
                    _, loss, _train_step, summaries_p = generator.train_step(sess, batch, [summaries_genr], global_step)
                    summary_writer.add_summary(summaries_p, _train_step)
                    running_avg_loss = _RunningAvgLoss(loss, running_avg_loss, summary_writer, _train_step)

                    if np.random.random() < 0.001:
                        idx = random.randint(0, batch.target_batch.shape[0] - 2)
                        src_ = batch.enc_batch[idx:idx+2]
                        src_lens_ = batch.enc_lens[idx:idx+2]
                        src_mask_ = batch.enc_padding_mask[idx:idx+2]
                        trg_ = batch.target_batch[idx:idx+2]
                        dec_ = batch.dec_batch[idx:idx+2]
                        if hps_train.pointer_gen:
                            oov_ = batch.art_oovs[idx:idx+2]

                        sample_feed = {generator.src_seq_batch: src_, generator.seq_len_batch: src_lens_,
                                       generator.enc_padding_mask: src_mask_}
                        if hps_train.pointer_gen:
                            sample_feed.update(
                                {generator.enc_batch_extend_vocab: batch.enc_batch_extend_vocab[idx:idx+2],
                                 generator.max_art_oovs: batch.max_art_oovs})
                        sample_output = sess.run(generator.gen_x, sample_feed)
                        for i in range(2):
                            art_words = data.outputids2words(
                                src_[i], vocab, (oov_[i] if hps_train.pointer_gen else None))
                            trg_words = data.outputids2words(
                                trg_[i], vocab, (oov_[i] if hps_train.pointer_gen else None))
                            dec_words = data.outputids2words(
                                dec_[i], vocab, (oov_[i] if hps_train.pointer_gen else None))
                            print ' '.join(art_words)
                            print ' '.join(trg_words)
                            print ' '.join(dec_words)
                            try:
                                out_words = data.outputids2words(
                                    sample_output[i], vocab, (oov_[i] if hps_train.pointer_gen else None))
                                print ' '.join(out_words)
                            except Exception as excp:
                                print excp.args
                                print '%s '%(oov_[i])
                                print '%s '%(sample_output[i])
                            print '\n'
                                # print '%s '%(src_[i], trg_[i], dec_[i], sample_output[i])
                                # raise
                    if sess.run(global_step) >= minimum_eval_iteration and sess.run(global_step) % 2000 == 0:
                        eval_batcher = Batcher(eval_data_path, vocab, hps_test, single_pass=True, num_epoch=1)
                        searcher.replace_search_w_generator(sess, ph_gvars_sc, srch_rplc)
                        rouge_out, _, _, _ = rouge_eval_bsearch(sess, generator, searcher, eval_batcher, vocab,
                                                             rouge_ref_dir,
                                                             rouge_dec_dir, log_dir, hps_test)
                        # curr_r1_recall = rouge_out["rouge_1_recall"]
                        curr_r2_recall = rouge_out["rouge_2_recall"]
                        rouge_text = 'Eval: ' + ' '.join(
                            ["%s: %.4f " % (key.replace('rouge_',''), val) for key, val in rouge_out.items()])
                        print 'train epoch: ', epoch, 'rouge: ', rouge_text
                        buffer = 'epoch: %d iteration: %d rouge: %s\n'%(epoch, sess.run(global_step), rouge_text)
                        log.write(buffer)

                        if best_rouge_score == 0. or curr_r2_recall > best_rouge_score:
                            tf.logging.info('Found new best model with %.3f rouge2 recall. Saving to %s',
                                            curr_r2_recall, bestmodel_save_path)
                            ckpt_prev_best = tf.train.latest_checkpoint(eval_dir, 'checkpoint_best')
                            tf.logging.info('Saving model to %s', bestmodel_save_path)
                            save_fl = saver.save(sess, bestmodel_save_path, global_step=sess.run(global_step),
                                                 latest_filename='checkpoint_best')
                            ckpt_new_best = tf.train.latest_checkpoint(eval_dir, 'checkpoint_best')
                            ckpt_updated = list(set(saver.last_checkpoints) - {ckpt_new_best})
                            if ckpt_prev_best is not None:
                                ckpt_updated.append(ckpt_prev_best)
                            saver.recover_last_checkpoints(ckpt_updated)
                            tf.logging.info("saved train session... %s" % save_fl)
                            best_rouge_score = curr_r2_recall
                            sess.run(assign_bestscore, feed_dict={best_score_ph: best_rouge_score})

                            # evaluate on test data
                            test_batcher = Batcher(test_data_path, vocab, hps_test, single_pass=True, num_epoch=1)
                            rouge_out, _, _, _ = rouge_eval_bsearch(sess, generator, searcher, test_batcher, vocab,
                                                                 rouge_ref_dir,
                                                                 rouge_dec_dir, log_dir, hps_test)
                            rouge_text = 'Test: ' + ' '.join(
                                ["%s: %.4f " % (key.replace('rouge_',''), val) for key, val in rouge_out.items()])
                            print 'train epoch: ', epoch, 'rouge: ', rouge_text
                            buffer = 'epoch: %d iteration: %d rouge: %s\n' % (epoch, sess.run(global_step), rouge_text)
                            log.write(buffer)
                summary_writer.flush()
                t1 = time.time()
                tf.logging.info('seconds for training epoch: %.3f', t1 - t0)
                tf.logging.info('Generator loss: %f', loss)  # print the loss to screen

            eval_batcher = Batcher(eval_data_path, vocab, hps_test, single_pass=True, num_epoch=1)
            searcher.replace_search_w_generator(sess, ph_gvars_sc, srch_rplc)
            rouge_out, gseq_lst, gdin_lst, _ = rouge_eval_bsearch(
                sess, generator, searcher, eval_batcher, vocab, rouge_ref_dir, rouge_dec_dir, log_dir, hps_test)
            # curr_r1_recall = rouge_out["rouge_1_recall"]
            curr_r2_recall = rouge_out["rouge_2_recall"]
            rouge_text = 'ROUGE: ' + ' '.join(
                ["%s: %.4f " % (key.replace('rouge_',''), val) for key, val in rouge_out.items()])
            buffer = 'epoch: %d iteration: %d rouge: %s\n' % (epoch, sess.run(global_step), rouge_text)
            log.write(buffer)

            if best_rouge_score == 0. or curr_r2_recall > best_rouge_score:
                tf.logging.info('Found new best model with %.3f rouge2 recall. Saving to %s',
                                curr_r2_recall, bestmodel_save_path)
                ckpt_prev_best = tf.train.latest_checkpoint(eval_dir, 'checkpoint_best')
                tf.logging.info('Saving model to %s', bestmodel_save_path)
                save_fl = saver.save(sess, bestmodel_save_path, global_step=sess.run(global_step),
                                     latest_filename='checkpoint_best')
                ckpt_new_best = tf.train.latest_checkpoint(eval_dir, 'checkpoint_best')
                ckpt_updated = list(set(saver.last_checkpoints) - {ckpt_new_best})
                if ckpt_prev_best is not None:
                    ckpt_updated.append(ckpt_prev_best)
                saver.recover_last_checkpoints(ckpt_updated)
                tf.logging.info("saved train session... %s" % save_fl)
                best_rouge_score = curr_r2_recall
                sess.run(assign_bestscore, feed_dict={best_score_ph: best_rouge_score})

                # evaluate on test data
                test_batcher = Batcher(test_data_path, vocab, hps_test, single_pass=True, num_epoch=1)
                rouge_out, _, _, _ = rouge_eval_bsearch(sess, generator, searcher, test_batcher, vocab,
                                                     rouge_ref_dir,
                                                     rouge_dec_dir, log_dir, hps_test)
                rouge_text = 'Test: ' + ' '.join(
                    ["%s: %.4f " % (key.replace('rouge_',''), val) for key, val in rouge_out.items()])
                print 'train epoch: ', epoch, 'rouge: ', rouge_text
                buffer = 'epoch: %d iteration: %d rouge: %s\n' % (epoch, sess.run(global_step), rouge_text)
                log.write(buffer)

    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()
    log.close()


if __name__ == '__main__':
    parser = setup_argparser()
    args = parser.parse_args()
    exp_instance_dir = args.experiment_dir
    if exp_instance_dir is None:
        print "USAGE: python teaser_gen.py -e <experiment_directory> --pointer"
        sys.exit()
    if not os.path.exists(exp_instance_dir):
        os.mkdir(exp_instance_dir)

    configuration = {}
    configuration.update(
        {
            'EPOCH_NUM_GEN': 16,
            'train_data_path': [os.path.join(os.getcwd(), 'resources', 'dataset/corpus_baseline/train.bin')],
            'vocab_path': [os.path.join(os.getcwd(), 'resources', 'dataset/corpus_baseline/vocab')],
            'eval_data_path': [os.path.join(os.getcwd(), 'resources', 'dataset/corpus_baseline/val.bin')],
            'test_data_path': [os.path.join(os.getcwd(), 'resources', 'dataset/corpus_baseline/test.bin')],
            # 'train_data_path': [
            #     os.path.join(os.getcwd(), 'resources', 'dataset/bin_tsr_trlt35/train.bin'),
            #     os.path.join(os.getcwd(), 'resources', 'dataset/bin_tsr_trlt45/train.bin'),
            #     os.path.join(os.getcwd(), 'resources', 'dataset/bin_tsr_trlt55/train.bin'),
            #     os.path.join(os.getcwd(), 'resources', 'dataset/bin_tsr_trlt99/train.bin'),
            #     ],
            'EMB_DIM': [100],  # 32 # embedding dimension
            'HIDDEN_DIM': [200],  # 32 # hidden state dimension of lstm cell
            'DEC_SEQ_LENGTH': 25,  # sequence length
            'ENC_SEQ_LENGTH': 100,  # sequence length
            'SEED': 88,
            'BATCH_SIZE': [32],  # 64  # 80
            'LEARNING_RATE': [0.0007],  # 0.001
            'BEAM_SIZE': 4,
            'VOCAB_SIZE': [20000],
            'grad_clip': [50.],
            'rand_unif_init_mag': [0.1],
            'trunc_norm_init_std': 1e-4,
            'rand_norm_init_std': 0.1
        })

    config_list = list(ParameterGrid({k: v if type(v) == list else [v] for k, v in configuration.items()}))
    minimum_eval_iteration = 6000
    for i, config in enumerate(config_list):
        #########################################################################################
        #  Generator  Hyper-parameters
        ######################################################################################
        EPOCH_NUM_GEN = config['EPOCH_NUM_GEN']
        train_data_path = config['train_data_path']
        vocab_path = config['vocab_path']
        eval_data_path = config['eval_data_path']
        test_data_path = config['test_data_path']
        EMB_DIM = config['EMB_DIM']
        HIDDEN_DIM = config['HIDDEN_DIM']
        DEC_SEQ_LENGTH = config['DEC_SEQ_LENGTH']
        ENC_SEQ_LENGTH = config['ENC_SEQ_LENGTH']
        SEED = config['SEED']
        BATCH_SIZE = config['BATCH_SIZE']
        LEARNING_RATE = config['LEARNING_RATE']
        BEAM_SIZE = config['BEAM_SIZE']
        VOCAB_SIZE = config['VOCAB_SIZE']

        rand_unif_init_mag = config['rand_unif_init_mag']
        trunc_norm_init_std = config['trunc_norm_init_std']
        rand_norm_init_std = config['rand_norm_init_std']
        grad_clip = config['grad_clip']

        log_dir = os.path.join(exp_instance_dir, 'log_%.1f'%i)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        rouge_ref_dir = os.path.join(log_dir, 'ref')
        rouge_dec_dir = os.path.join(log_dir, 'dec')
        main()
