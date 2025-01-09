# Laboratory : library and tools for implementing and training neural networks.

### Description

*Laboratory* is a Python library workspace for :

* Executing and test neural network
 * Generating visual results
 * Computing metrics
 * these tools are Torch-independent
 * **These tools / components should be generic and have carefully standardized inputs / outputs !**

* Training neural networks with PyTorch
 * <Laboratory>/detector/core/modules/ (to change to torch_modules) should gather an ordered set/pool or Torch modules with common and standardized inputs / outputs that can be used as basic building blocks for constructing object detection models

Ex : different backbones, different detection heads / modules can be implemented, different losses can be implemented. A detection network is then constructed by "plugging" / putting these modules together.


### Installation

Laboratory and almost all its Python dependencies can be installed directly with pip3 as it is structured as a Python package. 

Other dependecies : OpenCV3

<pre>
pip3 install -e <path_to_Laboratory>
</pre>

### Coding and testing rules

TODO

### Examples

For use examples of Laboratory library, you can check :

* working_nb.ipynb

