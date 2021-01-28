# CNN/Daily Mail Reading Comprehension Task

A Pytorch implementation of this paper:

Chen D, Bolton J, Manning C D. A thorough examination of the cnn/daily mail reading comprehension task.(https://arxiv.org/pdf/1606.02858v2.pdf)


##  Getting Started

These instructions will get you running the codes of this model.

### Dependencies

* Python 3.8
* Pytorch 1.7.0
* scikit-learn 0.23.2

### Datasets

* The two processed RC datasets CNN and Daily Mail get from https://cs.nyu.edu/~kcho/DMQA/.
* The Word embeddings glove.6B get from http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip.

### Code Struture

```
|__ dataset/
        |__ cnn/ --> Datasets for cnn news
            |__ question/ --> Processed datasets 
            	|__ training/ --> training  dataset folder
            	|__ test/ --> testing dataset folder
            	|__ validation/ --> validation dataset folder
|__ main.py/ Codes for main program
|__ model.py/ Codes for attentive reader model from this paper
|__ train.py/ Codes for training/evaluating model
|__ utils.py/ Codes for reading and process the dataset
```

### Training

```
python main.py --train_date_path [path]
	       --test_date_path [path]
	       --glove_path [path]
```

### Reference

```
    @inproceedings{chen2016thorough,
        title={A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task},
        author={Chen, Danqi and Bolton, Jason and Manning, Christopher D.},
        booktitle={Association for Computational Linguistics (ACL)},
        year={2016}
    }
```

### Reference Code

https://github.com/danqi/rc-cnn-dailymail
