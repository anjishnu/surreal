# Surreal
Surreal is a Deep Learning based <i><b>su</b>per <b>re</b>solution <b>al</b>gorithm</i>

The goal for this project is to provide a generic super resolution algorithm for upscaling images. The first implementation is going to use a conditional DCGAN (Deep Convolutional Generative Adversarial Network) so that the generative model operates on perceptual loss - not some kind of hacky euclidean distance metric.

<b>Todos</b>
- Get DCGAN working on MNIST
- Get super-resolution working for MNIST
- Adapt super-resolution code for an RGB dataset
- Easy to use data-generator API which is dataset independent. 

<b>Done</b>
- Modularize and make the model definition code more object-oriented to make future modifications easier. 

<b>References</b>
- <i>Radford, A., Metz, L. and Chintala, S., 2015. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.</i>
- <i>Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z. and Shi, W., 2016. Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint arXiv:1609.04802.</i>
