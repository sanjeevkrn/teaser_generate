import sys
import os
import struct
import subprocess
import collections

import cPickle
from tensorflow.core.example import example_pb2
from glob import glob
from picklable_itertools.extras import equizip
import numpy as np

rng = np.random.RandomState(1234)

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000


def chunk_file(set_name, out_dir, chunk_size):
    in_file = os.path.join(out_dir, set_name + '.bin')
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(chunk_size):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(chunks_dir, chunks_names, out_dir, chunk_size):
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in chunks_names:
        print "Splitting %s data into chunks..." % set_name
        chunk_file(set_name, out_dir, chunk_size)
    print "Saved chunked data in %s" % chunks_dir


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print "Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir)
    stories = os.listdir(stories_dir)
    # make IO list file
    print "Making list of files to tokenize..."
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print "Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir)
    subprocess.call(command)
    print "Stanford CoreNLP Tokenizer has finished."
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


# def get_art_abs(story_file):
#   lines = read_text_file(story_file)
#
#   # Lowercase everything
#   lines = [line.lower() for line in lines]
#
#   # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
#   lines = [fix_missing_period(line) for line in lines]
#
#   # Separate out article and abstract sentences
#   article_lines = []
#   highlights = []
#   next_is_highlight = False
#   for idx,line in enumerate(lines):
#     if line == "":
#       continue # empty line
#     elif line.startswith("@highlight"):
#       next_is_highlight = True
#     elif next_is_highlight:
#       highlights.append(line)
#     else:
#       article_lines.append(line)
#
#   # Make article into a single string
#   article = ' '.join(article_lines)
#
#   # Make abstract into a signle string, putting <s> and </s> tags around the sentences
#   abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])
#
#   return article, abstract


def write_to_bin(in_file_names, out_file, makevocab=False, bin_size=100000):
    assert len(in_file_names) >= 2
    print "Making bin file for lines in %s and %s" % (in_file_names[0], in_file_names[1])

    article_fn = [fn for fn in in_file_names if 'article' in os.path.basename(fn)][0]
    title_fn = [fn for fn in in_file_names if 'title' in os.path.basename(fn)][0]
    if makevocab:
        vocab_counter = collections.Counter()

    file_size = sum([1 for l in open(in_file_names[0])])
    with open(out_file, 'wb') as writer:
        for idx, (article, abstract_) in enumerate(equizip(open(article_fn), open(title_fn))):
            if idx % bin_size == 0:
                print "Writing %i of %i; %.2f percent done" % (idx, file_size, float(idx) * 100.0 / float(file_size))
            abstract = SENTENCE_START + ' ' + abstract_ + ' ' + SENTENCE_END
            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.strip().lower()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.strip().lower()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.strip().lower().split(' ')
                abs_tokens = abstract.strip().lower().split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print "Finished writing file %s\n" % out_file

    # write vocab to file
    if makevocab:
        out_dir = os.path.dirname(out_file)
        print "Writing vocab file..."
        with open(os.path.join(out_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print "Finished writing vocab file"


def prepare_emb_vocab(out_dir, glove_file):
    _word_to_emb = {}
    glove_type = int(os.path.basename(glove_file).split('.')[-2][:-1])
    with open(glove_file, 'r') as glove_f:
        _count = 0
        for line in glove_f:
            pieces = line.strip().split()
            if len(pieces) != glove_type + 1:
                print 'Warning: incorrectly formatted line in embedding file: %s\n' % line
                continue
            _word_to_emb[pieces[0]] = np.array(pieces[1:], dtype='float32')
            _count += 1
    cPickle.dump(_word_to_emb, open(os.path.join(out_dir, 'emb_vocab.pkl'), 'wb'))
    print "Finished constructing embedding vocab of %i total words." % _count


def preprocess(in_file_fn, out_art_fn, out_ttl_fn):
  with open(in_file_fn) as infile:
    with open(out_art_fn, 'wb') as out_art:
      with open(out_ttl_fn, 'wb') as out_ttl:
        for ln in infile:
          itms = ln.strip().split('|||')
          if len(itms) == 2:
            out_art.write('%s\n'%itms[0])
            out_ttl.write('%s\n'%itms[1])
          else:
            out_art.write('%s\n'%' '.join(itms[:-1]))
            out_ttl.write('%s\n'%itms[-1])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "USAGE: python make_datafiles.py <giga_input_dir> <giga_output_dir>"
        sys.exit()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # Check input files exist
    if not os.path.isdir(in_dir):
        raise ValueError('Given folder %s does not exist' % in_dir)

    train_files = glob(os.path.join(in_dir, '*train*.txt'))
    valid_files = glob(os.path.join(in_dir, '*valid*.txt'))
    # valid_files = glob(os.path.join(in_dir, '*eval*.txt'))
    test_files = glob(os.path.join(in_dir, '*test*.txt'))

    # Create output directories
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Read the tokenized articles and abstracts, do a some postprocessing then write to bin files
    write_to_bin(test_files, os.path.join(out_dir, "test.bin"))
    write_to_bin(valid_files, os.path.join(out_dir, "val.bin"))
    write_to_bin(train_files, os.path.join(out_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    # chunks_dir = os.path.join(out_dir, "chunked")

    # chunks_names = ['test']#, 'val', 'test']
    # chunks_names = ['val']#, 'val', 'test']
    # chunk_all(chunks_dir, chunks_names, out_dir, chunk_size=1000)

    # chunks_names = ['train']  # , 'val', 'test']
    # chunk_all(chunks_dir, chunks_names, out_dir, chunk_size=100000)
    # prepare_emb_vocab(out_dir, glove_file='../glove.6B.100d.txt')
