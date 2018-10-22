# Video caption generation with Encoder-Decoder model

In this project, we developed basic **Encoder-Decoder** model, and **S2VT** model to generate video captions. In addition,
we also applied **Attention** machnism to improve performance.

## Getting Started

The following instructions will get you a copy of the project and running on your local machine for testing purposes.

### Prerequisite & Toolkits

The following are some toolkits and their version you need to install for running this project

* [Python 3.6](https://www.python.org/downloads/release/python-360/) - The Python version used
* [Pytorch 0.3](http://pytorch.org/) - Deep Learning for Python
* [Pandas 0.21.0](https://pandas.pydata.org/) - Data Analysis Library for Python

In addition, it is required to use **GPU** to run this project.

## Model Structures

The following are the model structures we implemented in Pytorch from scratch:
* [Baseline Model]
![image](https://github.com/CKRC24/Seq2seq-on-video-caption-generation/blob/master/baseline.PNG)
* [S2VT Model]
![image](https://github.com/CKRC24/Seq2seq-on-video-caption-generation/blob/master/S2VT.PNG)

In order to improve performance, we also implemented **Bahdanau Attention** and **Luong Attention**
* [Attention]
![image](https://github.com/CKRC24/Seq2seq-on-video-caption-generation/blob/master/attention.PNG)

## Reference
[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly
learning to align and translate. arXiv:1409.0473 <br/>
[2] Minh-Thang Luong, Hieu Pham, Christopher D. Manning. 2015. Effective Approaches to Attention-based
Neural Machine Translation <br/>
[3] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer. 2015. Scheduled Sampling for Sequence
Prediction with Recurrent Neural Networks <br/>
[4] Natsuda Laokulrat, Sang Phan, Noriki Nishida. 2016. Generating Video Description using
Sequence-to-sequence Model with Temporal Attention <br/>

## Authors

* [Brian Chiang](https://github.com/CKRC24)
* [Sean Lee](https://github.com/sam961124)
* [David Lai](https://github.com/dav1a1223)
