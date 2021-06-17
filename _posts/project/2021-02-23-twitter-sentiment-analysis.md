---
layout: post
title: Mind Reading— Analysis on the Most Effective algorithm for Natural Language Processing (NLP) Sentiment Analysis
description: >
    The project identifies the reason why BERT is a powerful tool for sentimental analysis in Twitter.
sitemap: false
hide_last_modified: true
image: /assets/img/project/2021-02-23-twitter-sentiment-analysis/2021-02-23-twitter-sentiment-analysis.jpg
category: [project]
tags: [machine-learning, Natural language processing]
---

Twitter is an American microblogging and social networking service on which users post and interact with messages known as "tweets". It is a global community of sharing information not only for individual users but also for businesses aiming to increase their brand awareness, boost sales, partner up with influencers, and manage reputation. With the advent of artificial intelligence, businesses are now able to perform sentiment analyis, keeping track of what is being said about their product and detect negative mentions before they become major issues.

## Abstract

The project identifies the reason why BERT is a powerful tool for sentimental analysis. By comparing BERT with two baseline models (traditional CNN and BiLSTM), it is concluded that BERT is a powerful neural network for gathering subjectinve information from text. The area of its application is endless.

## Technologies Used
- Google Colab Jupyter Notebook with GPU Acceleration 
- Python (3.9.2)
    - PyTorch (1.8.0)
        - torchtext (0.9.0)
- BERT

&nbsp;

# Full Text

[Click here to view the the text in GitHub.](https://github.com/chan030609/twitter-sentiment-analysis)

&nbsp;

## 1. Introduction
Sentiment analysis signifies the systamatic use of natural language processing (NLP) to study subjective information in an expression, such as emotions and attitudes. Expressions are classified as positive, negative, and neutral depending on their polarity values. In order to compare the performances of the different structures of neural networks, three types of deep neural networks were employed: CNN (Convolutional Neural Network), BiLSTM (Bidirectional Long Short-Term Memory), and BERT (Bidirectional Encoder Representations from Transformers). They were trained and tested on an archived Tweet dataset 'Sentiment140', which is accessible through Kaggle.

&nbsp;

&nbsp;

## 2. Related Work

&nbsp;

### BERT
BERT is Google's neural network-based technique for NLP. It was created and published in 2018 by Jacob Devlin and his colleagues from Google. The deep bidirectional nature of this architecture allows a neural network model to run inputs in multiple directions, one from past to future and another from future to test. Doing so, the model quickly gathers a greater amount of contextual information than non-bidirectional models.

&nbsp;

### Sentiment140
The model was trained on a static dataset 'Sentiment140', which can be obtained from [Kaggle website](https://www.kaggle.com/kazanova/sentiment140). This dataset is a csv file with six columns: target, id, date, flag, user, and text. 
Since there are about a million rows in this dataset, only 100,000 rows were extracted as a compacted sample. 

&nbsp;

&nbsp;

## 3. Approach

&nbsp;

### Baseline Model

There are two baseline models to be created: a convolutional neural network (CNN) and bidirectional long short-term memory (BiLSTM). For CNNs, once features pass thorugh a convolutional layer, they are abstracted to a feature map with a certain shape. Then, the network organizes the incrementally available data into feature-driven super-classes and improves upon existing hierarchical CNN models. This makes CNNs especially suitable for image classification tasks because CNNs effectively reduce the high dimensionality comrpised of image pixels. 

<img src="https://miro.medium.com/max/1262/0*Q9QTGlHsQjzIlhBw.png" height=300>{:.lead width="800" height="100" loading="lazy"}

&nbsp;

Another baseline model employs a bidirectional long short-term memory (BiLSTM) structure. The purpose of this model is to implement and mimic the bidirectional structure of BERT. BiLSTM is an extended version of LSTM. LSTM is an artificial recurrent neural network (RNN) architecture with speical feedback connections. Unlike traditional RNNs, LSTM units include a 'memory cell' that can preserve information for long periods of time. BiLSTMs go through the same process as LSTM models except that instead of processing data in one direction, they connect two hidden layers of opposite directions to the same output, yielding more accurate results than traditional LSTMs.

<img src="https://www.i2tutorials.com/wp-content/media/2019/05/Deep-Dive-into-Bidirectional-LSTM-i2tutorials.jpg" height=300>{:.lead width="800" height="100" loading="lazy"}

&nbsp;

Both CNN and BiLSTM were applied to pre-trained 100-dimensional word embeddings obtained from GloVe. Then, they were trained on `Sentiment140` dataset using a GPU Accelerator provided by Google Colab.

&nbsp;

### BERT

<img src="https://miro.medium.com/max/876/0*ViwaI3Vvbnd-CJSQ.png" height=300>{:.lead width="800" height="100" loading="lazy"}

BERT created by Google was established upon the layers of BiLSTMs, inherinting their multidirectional nature. But in BERT, the model processes data from all possible positions instead of relying on two directions, which might take longer but usually yield better results. In this project, BERT was used to compare its performance with the two baseline models mentioned above. The comparison was conducted by calculating and comparing the accuracies of their predictions on the test dataset. 

&nbsp;

&nbsp;

## 4. Project Details
&nbsp;

For more code details, click _"Open in Colab"_.

&nbsp;

### [Baseline 1](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/cnn.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/cnn.ipynb)


The first baseline model to be experimented was CNN. The batch size was consistently 128.

After 5 epochs of training, both the valdiation accuracy and test accuracy were around 79%. Each epoch took about 2 minutes to complete.
```
[Train Result]
Epoch: 01 | Epoch Time: 1m 55s
	Train Loss: 0.504 | Train Acc: 75.00%
	 Val. Loss: 0.441 |  Val. Acc: 79.51%
Epoch: 02 | Epoch Time: 1m 56s
	Train Loss: 0.413 | Train Acc: 81.52%
	 Val. Loss: 0.435 |  Val. Acc: 80.43%
Epoch: 03 | Epoch Time: 1m 55s
	Train Loss: 0.359 | Train Acc: 84.44%
	 Val. Loss: 0.440 |  Val. Acc: 80.36%
Epoch: 04 | Epoch Time: 1m 55s
	Train Loss: 0.308 | Train Acc: 86.95%
	 Val. Loss: 0.475 |  Val. Acc: 79.89%
Epoch: 05 | Epoch Time: 1m 55s
	Train Loss: 0.261 | Train Acc: 89.27%
	 Val. Loss: 0.550 |  Val. Acc: 78.82%
```
```
[Test Result]
Test Loss: 0.45 | Test Acc: 79.47%
```

&nbsp;

### [Baseline 2](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/bilstm.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/bilstm.ipynb)

The second baseline model to be experimented was BiLSTM. The batch size was consistently 128.

After 5 epochs of training, both the validation accuracy and the test accuracy were around 81%. Each epoch took about 10 seconds to complete.
```
[Train Result]
Epoch 1:
	 Total Time: 0m 10s
	 Train Loss 0.51 | Train Accuracy: 75.08%
	 Validation Loss 0.43 | Validation Accuracy: 80.28%
Epoch 2:
	 Total Time: 0m 10s
	 Train Loss 0.4 | Train Accuracy: 81.98%
	 Validation Loss 0.41 | Validation Accuracy: 81.52%
Epoch 3:
	 Total Time: 0m 10s
	 Train Loss 0.35 | Train Accuracy: 84.77%
	 Validation Loss 0.42 | Validation Accuracy: 81.48%
Epoch 4:
	 Total Time: 0m 10s
	 Train Loss 0.31 | Train Accuracy: 86.96%
	 Validation Loss 0.42 | Validation Accuracy: 81.46%
Epoch 5:
	 Total Time: 0m 11s
	 Train Loss 0.27 | Train Accuracy: 88.62%
	 Validation Loss 0.43 | Validation Accuracy: 81.47%
```
```
[Test Result]
Test Loss: 0.41 | Test Acc: 82.05%
```

The two baseline models above were experimented under the same condition. Although both models had similar accuracy, the time it took each epoch was significantly different, BiLSTM being about 10 times faster than CNN. It is concluded that BiLSTM is better designed for NLP than CNN because of its forward-backward feedbacks.
{:.note}

 &nbsp;

### [BERT](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/bert.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chan030609/twitter-sentiment-analysis/blob/master/bert.ipynb)

The final model to be experimented was BERT. The batch size was consistently 128.

After 5 epochs of training, both the validation accuracy and the test accuracy were around 82%. Each epoch took about 5 minutes to complete.

```
[Train Result]
Epoch: 01 | Epoch Time: 5m 3s
	Train Loss: 0.481 | Train Acc: 76.65%
	 Val. Loss: 0.415 |  Val. Acc: 80.78%
Epoch: 02 | Epoch Time: 5m 7s
	Train Loss: 0.415 | Train Acc: 81.13%
	 Val. Loss: 0.402 |  Val. Acc: 82.18%
Epoch: 03 | Epoch Time: 5m 8s
	Train Loss: 0.392 | Train Acc: 82.35%
	 Val. Loss: 0.395 |  Val. Acc: 82.11%
Epoch: 04 | Epoch Time: 5m 7s
	Train Loss: 0.371 | Train Acc: 83.57%
	 Val. Loss: 0.395 |  Val. Acc: 82.77%
Epoch: 05 | Epoch Time: 5m 8s
	Train Loss: 0.350 | Train Acc: 84.59%
	 Val. Loss: 0.405 |  Val. Acc: 83.01%
```
```
[Test Result]
Test Loss: 0.386 | Test Acc: 82.79%
```
BERT turned out much less time efficient than the second baseline model, but this time the accuracy has slightly improved compared to the regular BiLSTM model.
{:.note}

 &nbsp;

## 5. Conclusion

Sentimental analyis is a field of study that analyzes people's sentiments, attitudes, or emotions toward certain topics. Among many pre-trained NLP models, BERT proves to be especially powerful in the field of sentimental analysis with its multidirectional feedbacks and a complex structure of layers. In fact, the BERT algorithm is already being used to help Google Search better interpret what users are asking. Considering its high performance, BERT has clear advantages in sentiment analysis. With the advent of the big data technologies, perhaps BERT is a kind of technology we have been awaited the most.
