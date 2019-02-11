# Image Segmentation 

* original source: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite.git

![](https://github.com/merrybingo/Semantic-Segmentation-Suite/blob/master/Images/semseg.gif)



## Environment

* Debian
  * Google Cloud Platform
  * image:  c2-deeplearning-tf-1-12-cu100-20190125b2
    * Google, Intel® optimized Deep Learning Image: TensorFlow 1.12.0, m19 (with Intel® MKL-DNN/MKL and CUDA 10.0), A **Debian** based image with TensorFlow (With CUDA 10.0 and Intel® MKL-DNN, Intel® MKL) plus Intel® optimized NumPy, SciPy, and scikit-learn.
  * 머신 유형: n1-standard-4(vCPU 4개, 15GB 메모리)
  * CPU 플랫폼: Intel Ivy Bridge
  * GPU: 1 x NVIDIA Tesla K80

* Anaconda 3

* Python 3.7

```bash
conda create -n segmentation python=3.7
conda activate segmentation
```



## Dependencies

* tensorflow-gpu
* numpy
* opencv-python
* matplotlib
* scipy
* scikit-learn
* imageio



## Installation

```bash
pip install -r requirements.txt
```



## Usage

```bash
Python train.py
```

