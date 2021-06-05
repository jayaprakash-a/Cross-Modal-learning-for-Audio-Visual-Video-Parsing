# Cross-Modal-learning-for-Audio-Visual-Video-Parsing



This is the official code release for paper **"Cross-Modal learning for Audio-Visual Video Parsing"**. The paper is accepted and would be presented at [Interspeech 21](https://www.interspeech2021.org/)



Arxiv Link : https://arxiv.org/abs/2104.04598



# Instructions

### Setting up environment

Install Anaconda. More details [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Install required packages from the conda environment file 

```bash
conda env create -f environment.yml
```



### Downloading datasets

#### LLP dataset 

LLP dataset is a labelled subset of the parent AVVP dataset. The scripts for downloading and feature extraction can be obtained [here](https://github.com/YapengTian/AVVP-ECCV20). Please follow the instructions.

#### AVVP dataset for pre-training

You can download any amount of AudioSet clips from [here](https://research.google.com/audioset/download.html). For our research purposes we used only a part of it. The YouTube IDs of the same can be found in **data/....** file.

### Generating the ground truth for AVG

To obtain the ground truth for Audio visual grounding task. Please follow the following instructions

```bash
conda activate $env_name
python compute_gt.py --video_dir $PATH_TO_VIDEO_FEATS -audio_feats $PATH_TO_AUDIO_FEATS -gt_file $FILENAME_OF_COMPUTED_GT
```



### Training the Model

1. To train the simple HAN model follow instructions from [here](https://github.com/YapengTian/AVVP-ECCV20).
2. Training the adverserial network
```bash
python main_avvp_adv.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS
```
3. Training the global context aware network
```bash
python main_avvp_gca.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS
```
4. Training adverserial network with skip connections
```bash
python main_avvp_adv_skip.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS
```
5. Training the pre-training network end-to-end with uni-modal grounding objective
```bash
python main_avvp_uni.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS -gt_file $FILENAME_OF_COMPUTED_GT
```
6. Training the pre-training network end-to-end with cross-modal grounding objective
```bash
python main_avvp_cross.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS -gt_file $FILENAME_OF_COMPUTED_GT
```
7. Training the pre-training network end-to-end with multi-modal grounding objective
```bash
python main_avvp_multi.py --mode train --video_dir $PATH_TO_VIDEO_FEATS --audio_dir $PATH_TO_AUDIO_FEATS -gt_file $FILENAME_OF_COMPUTED_GT
```


**In case of any troubles, raise an issue or mail any of authors**

If you want to use our work, please cite : 

```tex
@article{jatin2021cmlavvp,
  author    = {Jatin Lamba and
               Abhishek and
               Jayaprakash Akula and
               Rishabh Dabral and
               Preethi Jyothi and
               Ganesh Ramakrishnan},
  title     = {Cross-Modal learning for Audio-Visual Video Parsing},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.04598}
}

```

### Acknowledgement

We are grateful to ......  for their support and sponsorship.
