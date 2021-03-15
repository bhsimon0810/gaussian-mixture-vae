# Gaussian Mixture VAE for Document Modeling
Tensorflow implementation of Gaussian mixture VAE (with Householder flow) for document modeling.  We assume that a document is modeled as a mixture of classes, and a class is modeled as a mixture of latent topics. The key idea is first to use a classifier to predict the probabilities with which the document is assigned to all the classes. Then this distribution will be considered as the mixture weights in the GMM.

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="https://github.com/bhsimon0810/gaussian-mixture-vae/blob/main/models/gmm_vae.PNG">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">Figure. The architecture of GMM-VAE (with Householder flow) </div> </center>

### Dependencies

- Python 3.6
- Tensorflow == 1.14.0

### Usage

#### Datasets

- The processed datasets (i.e. IMDB, DBPedia, AGNews, 20Newsgroup) can be downloaded [here](https://pan.baidu.com/s/1HQpAgLJ3tLJpZy_X1_8EbQ). (zc87) 
- The glove word embedding.

#### Training

To train the model (please check all the hyper-parameters)

```
python main.py
```

### Reference

1.  [Neural Variational Inference for Text Processing](https://arxiv.org/abs/1511.06038) 
2. [Variational Inference with Gaussian Mixture Model and Householder Flow](https://www.sciencedirect.com/science/article/abs/pii/S0893608018302879)

