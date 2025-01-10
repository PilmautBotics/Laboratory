# Laboratory : library and tools for implementing and training neural networks.

### Description

*Laboratory* is a Python library workspace for:

* Executing and test neural network
 * Generating visual results
 * Computing metrics
 * these tools are Torch-independent
 * **These tools / components should be generic and have carefully standardized inputs / outputs !**

* Training neural networks with PyTorch
 * <Laboratory>/detector/core/modules/ (to change to torch_modules) should gather an ordered set/pool or Torch modules with common and standardized inputs / outputs that can be used as basic building blocks for constructing detector models

Ex : different backbones, different detection heads / modules can be implemented, different losses can be implemented. A detection network is then constructed by "plugging" / putting these modules together.

It is organized as follow: 
Laboratory: 
---- Classificator : implementation for classification use cases
---- Detector : implementation for detection use cases
---- Segmentator : implementation for segmentation use cases

### Installation

Laboratory and almost all its Python dependencies can be installed directly with pip3 as it is structured as a Python package.

You can download Miniconda to install minimal anaconda environmnent via: https://docs.anaconda.com/miniconda/

Create environment  for development:
<pre>
conda create --name pytorch_env python=3.9.16
conda activate pytorch_env
pip install torch==1.13.1 torchvision==0.14.1 scikit-learn==1.2.1 torch-geometric==2.2.0
</pre>

### Coding and testing rules

TODO

