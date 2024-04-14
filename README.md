<div align="center">
<h1 align="center"> MapTracker: Tracking with Strided Memory Fusion for <br/> Consistent Vector HD Mapping </h1>



### [Jiacheng Chen*<sup>1</sup>](https://jcchen.me) , [Yuefan Wu*<sup>1</sup>](https://ivenwu.com/) , [Jiaqi Tan*<sup>1</sup>](https://christinatan0704.github.io/mysite/), [Hang Ma<sup>1</sup>](https://www.cs.sfu.ca/~hangma/), [Yasutaka Furukawa<sup>1,2</sup>](https://www2.cs.sfu.ca/~furukawa/)

### <sup>1</sup> Simon Fraser University <sup>2</sup> Wayve

### [arXiv](https://arxiv.org/abs/2403.15951), [Project page](https://map-tracker.github.io/)

</div>




https://github.com/woodfrog/maptracker_mock/assets/13405255/862d4be4-b8ff-4a0c-8439-176fbc92968c

This repository provides the official implementation of the paper [MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping](https://arxiv.org/abs/2403.15951). MapTracker reconstructs temporally consistent vector HD maps, and the local maps can be progressively merged into a global reconstruction.

This repository is built upon [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet). 


## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Introduction
This paper presents a vector HD-mapping algorithm that formulates the mapping as a tracking task and uses a history of memory latents to ensure consistent reconstructions over time.

Our method, MapTracker, accumulates a sensor stream into memory buffers of two latent representations: 1) Raster latents in the bird's-eye-view (BEV) space and 2) Vector latents over the road elements (i.e., pedestrian-crossings, lane-dividers, and road-boundaries). The approach borrows the query propagation paradigm from the tracking literature that explicitly associates tracked road elements from the previous frame to the current, while fusing a subset of memory latents selected with distance strides to further enhance temporal consistency. A vector latent is decoded to reconstruct the geometry of a road element.

The paper further makes benchmark contributions by 1) Improving processing code for existing datasets to produce consistent ground truth with temporal alignments and 2) Augmenting existing mAP metrics with consistency checks. MapTracker significantly outperforms existing methods on both nuScenes and Agroverse2 datasets by over 8% and 19% on the conventional and the new consistency-aware metrics, respectively.


## Model Architecture

![visualization](docs/fig/arch.png)

(Top) The architecture of MapTracker, consistsing of the BEV and VEC Modules and their memory buffers. (Bottom) The close-up views of the BEV and the vector fusion layers.

The **BEV Module** takes ConvNet features of onboard perspective images, the BEV memory buffer ${M_{\text{BEV}}(t-1), M_{\text{BEV}}(t-2),\ ... }$ and vehicle motions ${P^t_{t-1}, P^t_{t-2},\ ... }$ as input. It propagates the previous BEV memory $M_{\text{BEV}}(t-1)$ based on vehicle motion to initialize $M_{\text{BEV}}(t)$. In the BEV Memory Fusion layer, $M_{\text{BEV}}(t)$ is integrated with selected history BEV memories $\{M_{\text{BEV}}^{*}(t'), t'\in \pi(t)\}$, which is used for semantic segmentation and passed to the VEC Module.

The **VEC Module** propagates the previous latent vector memory $M_{\text{VEC}}(t-1)$ with a PropMLP to initialize the vector queries $M_{\text{VEC}}(t)$. In Vector Memory Fusion layer, each propagated $M_{\text{VEC}}(t)$ is fused with its selected history vector memories $\{M_{\text{VEC}}^{*}(t'), t' \in \pi(t)\}$. The final vector latents are decoded to reconstruct the road elements.


## Installation

Please refer to the [installation guide](docs/installation.md) to set up the environment.


## Data preparation

For how to download and prepare data for the nuScenes and Argoverse2 datasets, as well as downloading our checkpoints, please see the [data preparation guide](docs/data_preparation.md). 


## Getting Started

For instructions on how to run training, inference, evaluation, and visualization, please follow [getting started guide](docs/getting_started.md).


## Acknowledgements

We're grateful to the open-source projects below, their great work made our project possible:

* BEV perception: [BEVFormer](https://github.com/fundamentalvision/BEVFormer) ![GitHub stars](https://img.shields.io/github/stars/fundamentalvision/BEVFormer.svg?style=flat&label=Star)
* Vector HD mapping: [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) ![GitHub stars](https://img.shields.io/github/stars/yuantianyuan01/StreamMapNet.svg?style=flat&label=Star), [MapTR](https://github.com/hustvl/MapTR) ![GitHub stars](https://img.shields.io/github/stars/hustvl/MapTR.svg?style=flat&label=Star)


## Citation

If you find MapTracker useful in your research or applications, please consider citing:

```
@inproceedings{chen2024maptrakcer,
  author  = {Chen, Jiacheng and Wu, Yuefan and Tan, Jiaqi and Ma, Hang and Furukawa, Yasutaka},
  title   = {MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping},
  journal = {arXiv preprint arXiv:2403.15951},
  year    = {2024}
}
```

## License

This project is licensed under GPL, see the [license file](LICENSE) for details.
