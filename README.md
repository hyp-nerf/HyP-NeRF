# 🚀HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork
> [NeurIPS 2023](https://nips.cc/)

[Bipasha Sen](https://bipashasen.github.io/)* <sup>**1**</sup> [Gaurav Singh](https://vanhalen42.github.io/)* <sup>**1**</sup>, [Aditya Agarwal](https://skymanaditya1.github.io/)* <sup>**1**</sup>, [Rohith Agaram](https://scholar.google.com/citations?user=Ni6qG7wAAAAJ) <sup>**1**</sup>, [Madhava Krishna](https://scholar.google.com/citations?user=QDuPGHwAAAAJ) <sup>**1**</sup>, [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/) <sup>**2**</sup>

*denotes equal contribution, <sup>**1**</sup> International Institute of Information Technology Hyderabad, <sup>**2**</sup> Brown University


https://github.com/hyp-nerf/HyP-NeRF/assets/71246220/1ec44e00-5c2e-488c-8335-61fd911df801

This is the official implementation of the paper _"HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork"_ accepted at **NeurIPS 2023**

## 👉 TODO 
- [ ] Code Release
  - [x] Training Code
  - [x] Architecture modules, renderer, Meta MRHE
  - [x] Pretrained Compression Checkpoint
- [ ] ...

## CREATING THE ENVIRONMENT 
Please follow the steps outlined in [torch-ngp](https://github.com/ashawkey/torch-ngp#install) repository for creating the environment upto and including the `Build extension` subheading. 

Note: Please build the extensions using the source code on this repository. 

## Dataset
Download the [ABO Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html). We use the images and the transforms from [abo-benchmark-material.tar](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-benchmark-material.tar) and the metadata file [abo-listings.tar](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar) for training. Place them in a directory structure as follows:
```
dataset_root
├── ABO_rendered
│   ├── B00EUL2B16
│   ├── B00IFHPVEU
│   ...
│
└── ABO_listings
  └── listings
      └── metadata
          ├── listings_0.json.gz
          ...
          └── listings_f.json.gz
```
## Training
To train a model on the ABO Chair dataset run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path <dataset_root> --workspace <workspace dir> --bound 1.0 --scale 0.8 --dt_gamma 0 --class_choice CHAIR --load_ckpt
```

## Compression Demo

Download the pretrained compression checkpoint from [here](https://drive.google.com/file/d/1GFWLWh2waQtqdw8mcVOi2sag2NkJr997/view?usp=sharing) for the CHAIR category and place it in your workspace dir as follows:

```
<Workspace dir>
└── checkpoints
    └── ngp_ep<>.pth
```

To render a specific NeRF from the codebook, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --workspace <Workspace dir> --bound 1.0 --scale 0.6 --dt_gamma 0 --class_choice CHAIR --load_ckpt --test --test_index <index of codebook (max index 1037)>
```
## 👏 Acknowledgement

Some parts of the code are inspired and borrowed from [torch-ngp](https://github.com/ashawkey/torch-ngp) (which we use as our backbone) and [INR-V](https://github.com/bipashasen/INR-V-VideoGenerationSpace). We thank the authors for providing the source code.


## 📜 BibTeX

If you find HyP-NeRF useful in your work, consider citing us.
```
@article{hypnerf2023,
  title={HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork},
  author={Sen, Bipasha and Singh, Gaurav and Agarwal, Aditya and Agaram, Rohith and Krishna, K Madhava and Sridhar, Srinath},
  journal={NeurIPS},
  year={2023}
}
```




<!--
**hyp-nerf/HyP-NeRF** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
