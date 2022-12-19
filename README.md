<h1 align="center">
<p>Deep Learning (CSGY- 6923) :bar_chart:</p>
<p>NYU</p>

<p align="center">
<img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C3">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>Image Super Resolution through Energy Based Models</p>
</h2>



<h4 align="centre"> 
    <p align="centre" > High Res, Low Res and EBM output  </p>
    <img src="https://github.com/vaibhav016/Energy-Based-Models-for-Image-Resolution/blob/main/images/red_panda.png" width="700" height="200" />
</h4>


<h4 align="centre"> 
    <p align="centre" > High Res, Low Res and EBM output  </p>
    <img src="https://github.com/vaibhav016/Energy-Based-Models-for-Image-Resolution/blob/main/images/camera.png" width="700" height="200" />
</h4>

## Supervised by Prof. Chinmay Hegde and Prof Arslan Mosenia 

### Built by 
- Vaibhav Singh (vs2410)
- Sindhu Bhoopalam Dinesh (sb8019)
- Sourabh Kumar Bhattacharjee (skb5275)


<h4 align="centre"> 
    <p align="centre" > Typical Energy Surface Evolution </p>
    <img src="https://github.com/vaibhav016/Energy-Based-Models-for-Image-Resolution/blob/main/images/ebm.png" width="600" height="250" />
</h4>

## Table of Contents

<!-- TOC -->

- [Installation](#installation)  
- [Training](#Training)  
     - [EBMS](#EBMS)
     - [SRGAN](#SRGAN)
- [Inference](#Inference)
- [Colab Notebooks](#ColabNotebooks)
- [Training Dynamics](#TrainingLogs)


<!-- /TOC -->

### Installation

```bash
https://github.com/vaibhav016/Energy-Based-Models-for-Image-Resolution.git
cd Energy-Based-Models-for-Image-Resolution
```
1.) Installation on conda environment -  
```bash
conda env create --name v_env --file=environments.yml
```
2.) Installation via requirements.txt -
```bash
pip install requirements.txt
```
### Training 
For Training EBM based model 
```bash
python3 ebm.py -tr 1 
```
For Training and Inference of SRGAN model
```bash
python3 train_srgan.py 
```
For Training and Inference of SRGAN2 model
```bash
python3 train_srgan2.py 
```


<!---
python3 train_srgan.py -tr 1 -d "data_path" 
and here
-->

### Inference 
For Inference EBM based model 
```bash
python3 ebm.py -tr 0
```

### Colab Notebooks
The above commands are for deploying this project on a server or a cloud. 
We ran simultanous experiments on google colab and share the following notebooks for easier interpretability. The notebooks can be found in Jupyter Notebook Experiments directory. 

EBM Colab Notebook: (https://colab.research.google.com/drive/1Zg4LsIeLAQnVwEu8nTveQlL6RTefB6pW?usp=sharing)

SRGAN Colab Notebook: (https://colab.research.google.com/drive/1CeNWaI_EwDiMzXHtIzg1IN3BGjX3FDaM?usp=sharing)

### Training Dynamics 
<h4 align="centre"> 
    <p align="centre">EBM Contrastive Loss dynamics</p> 
    <img src="https://github.com/vaibhav016/Energy-Based-Models-for-Image-Resolution/blob/main/images/contrastive_loss.png" width="600" height="300" />
</h4>
