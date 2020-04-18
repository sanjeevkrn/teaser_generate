"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys

import tensorflow as tf
from tensorflow.core.example import example_pb2
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')
tf.app.flags.DEFINE_string('col_art', '', 'column for article')
tf.app.flags.DEFINE_string('col_abs', '', 'column for abstract')


def _binary_to_text():
    reader = open(FLAGS.in_file, 'rb')
    writer = open(FLAGS.out_file, 'w')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes:
            sys.stderr.write('Done reading\n')
            return
        str_len = struct.unpack('q', len_bytes)[0]
        tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(tf_example_str)
        examples = []
        for key in tf_example.features.feature:
            examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
        writer.write('%s\n' % '\t'.join(examples))
    reader.close()
    writer.close()


def _binary_to_text_v2():
    reader = open(FLAGS.in_file, 'rb')
    writer = open(FLAGS.out_file, 'w')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes:
            sys.stderr.write('Done reading\n')
            return
        str_len = struct.unpack('q', len_bytes)[0]
        tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(tf_example_str)
        examples = []
        for key in tf_example.features.feature:
            if key == 'article':
                art_old = tf_example.features.feature[key].bytes_list.value[0].replace('=', '')
                art = '<d> <p> <s> ' + art_old.decode('ascii', 'ignore') + ' </s> </p> </d>'
                examples.append('%s=%s' % (key, art))
            elif key == 'abstract':
                abs_old = str(tf_example.features.feature[key].bytes_list.value[0]).replace('=', '').replace('<s> ',
                                                                                                             '').replace(
                    ' </s>', '')
                abs = '<d> <p> <s> ' + abs_old.strip().decode('ascii', 'ignore') + ' </s> </p> </d>'
                examples.append('%s=%s' % (key, abs))
            else:
                raise ValueError('%s should not be possible, verify bin' % key)
        writer.write('%s\n' % '\t'.join(examples))
    reader.close()
    writer.close()


def _json_to_binary():
    writer = open(FLAGS.out_file, 'wb')
    for itm in open(FLAGS.in_file, 'r'):
        js_line = json.loads(itm)
        tf_example = example_pb2.Example()
        article_sentences = js_line['tokenized_article'].encode('utf-8', 'ignore')
        tweet = js_line['tweet'].encode('utf-8', 'ignore')
        tf_example.features.feature['article'].bytes_list.value.extend([article_sentences])
        tf_example.features.feature['abstract'].bytes_list.value.extend([tweet])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
    writer.close()


def _json_to_text():
    with open(FLAGS.out_file+'_article.txt', 'wb') as writer_article:
        with open(FLAGS.out_file+'_title.txt', 'wb') as writer_title:
            for itm in open(FLAGS.in_file, 'r'):
                js_line = json.loads(itm)
                article_sentences = js_line[FLAGS.col_art].encode('utf-8', 'ignore')
                tweet = js_line[FLAGS.col_abs].encode('utf-8', 'ignore')
                writer_article.write(article_sentences+'\n')
                writer_title.write(tweet+'\n')


def _text_to_binary():
    inputs = open(FLAGS.in_file, 'r').readlines()
    writer = open(FLAGS.out_file, 'wb')
    for inp in inputs:
        tf_example = example_pb2.Example()
        for feature in inp.strip().split('\t'):
            (k, v) = feature.split('=')
            tf_example.features.feature[k].bytes_list.value.extend([v])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
    writer.close()


def main(unused_argv):
    assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
    if FLAGS.command == 'binary_to_text':
        _binary_to_text()
    elif FLAGS.command == 'text_to_binary':
        _text_to_binary()
    elif FLAGS.command == 'binary_to_text_v2':
        _binary_to_text_v2()
    elif FLAGS.command == 'json_to_binary':
        _json_to_binary()
    elif FLAGS.command == 'json_to_text':
        _json_to_text()


if __name__ == '__main__':
    tf.app.run()
