## News Article Teaser Tweet Generator.
A teaser is a short reading suggestion for an article that is illustrative and includes curiosity-arousing elements to entice potential readers to read particular news items.
This code trains seq2seq models using a collection of news articles and teasers.

### Dependencies
* python 2.7
* tensorflow-gpu==1.13.1

### run experiments
This code is based on [TextSum code](https://github.com/tensorflow/models/tree/master/textsum) from Google Brain and [pointer-generator code](https://github.com/abisee/pointer-generator) from [See et al 2017](https://arxiv.org/abs/1704.04368).

To collect teasers, please check [this repository](https://github.com/sanjeevkrn/teaser_collect).

Download our NAACL 2019 train, eval and test teaser dataset:
https://s3-eu-west-1.amazonaws.com/teasers.naacl19/teaser_base_naacl19.tar.gz

```
cd teaser_generate/app/
cp path_to_teaser_base_naacl19.tar.gz resources/dataset/
tar -xvzf resources/dataset/teaser_base_naacl19.tar.gz -C resources/dataset/
```
Create source-target (article-title) pairs out of json files
```
python src/data_convert_example.py --command 'json_to_text' --in_file resources/dataset/corpus_baseline/up2mar19_train_new.shuf.json --out_file resources/dataset/corpus_baseline/up2mar19_train --col_art 'tokenized_article' --col_abs 'tokenized_tweet' --col_dmn 'kmeans_class'
python src/data_convert_example.py --command 'json_to_text' --in_file resources/dataset/corpus_baseline/up2mar19_test_new.json --out_file resources/dataset/corpus_baseline/up2mar19_test --col_art 'tokenized_article' --col_abs 'tokenized_tweet' --col_dmn 'kmeans_class'
python src/data_convert_example.py --command 'json_to_text' --in_file resources/dataset/corpus_baseline/up2mar19_valid_new.json --out_file resources/dataset/corpus_baseline/up2mar19_valid --col_art 'tokenized_article' --col_abs 'tokenized_tweet' --col_dmn 'kmeans_class'
```
Create tf bin files out of source-target files
```
python make_datafiles.py resources/dataset/corpus_baseline/ resources/dataset/corpus_baseline/
```
Train baseline vanilla seq2seq model
```
python src/teaser_gen.py -e resources/model_s2s_vanilla
```
Train baseline pointer seq2seq model
```
python src/teaser_gen.py -e resources/model_s2s_pointer --pointer
```  
#### For more details, check out our paper:
- SK Karn, M Buckley, U Waltinger, H Sch{\"u}tze. [News Article Teaser Tweets and How to Generate Them](https://www.aclweb.org/anthology/N19-1398.pdf).

## How do I cite this work?
```
@inproceedings{karn-etal-2019-news,
    title = "News Article Teaser Tweets and How to Generate Them",
    author = {Karn, Sanjeev Kumar  and
      Buckley, Mark  and
      Waltinger, Ulli  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1398",
    doi = "10.18653/v1/N19-1398",
    pages = "3967--3977",
    abstract = "In this work, we define the task of teaser generation and provide an evaluation benchmark and baseline systems for the process of generating teasers. A teaser is a short reading suggestion for an article that is illustrative and includes curiosity-arousing elements to entice potential readers to read particular news items. Teasers are one of the main vehicles for transmitting news to social media users. We compile a novel dataset of teasers by systematically accumulating tweets and selecting those that conform to the teaser definition. We have compared a number of neural abstractive architectures on the task of teaser generation and the overall best performing system is See et al. seq2seq with pointer network.",
}
```