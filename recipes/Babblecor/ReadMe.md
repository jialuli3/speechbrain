## BabbleCor: Baby Sounds Sub-Challenge in Interspeech 2019 Computational Paralinguistics Challenge 
This recipe is developed based on SpeechBrain toolkit. This recipe contains scripts for training and testing children's vocalization classifications using wav2vec 2.0 model pretrained on infant vocalizations and children ASR.

## Uses
### Install SpeechBrain
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .

```

### Download pretrained wav2vec2 models on LittleBeats and LENA audio ###

Our pretrained ASR model weights can be downloaded via our Hugging Face repository

https://huggingface.co/lijialudew/wav2vec_Providence

### Check out this branch
```
git clone https://github.com/jialuli3/speechbrain.git
cd speechbrain
git checkout -b infant-voc-classification
git pull origin infant-voc-classification
```

### Prepare data in json format ###
To make data compatible with this script, prepare your data similar as the following json format (each entry corresponds to each sample)
```
{
  "sample_data1": { 
    "wav": "path/to/your/wav",
    "label": "Junk", 
    },
  "sample_data2": { 
    "wav": "path/to/your/wav",
    "label": "Non-canonical", 
    },
}
```
Vocalization types include:
  - *Non-canonical*
  - *Canonical*
  - *Laugh*
  - *Crying*
  - *Junk*

### Make yaml files in *hparams* folder compatiable with your dataset
- Change data paths in *train_annotation*, *valid_annotation*, and *test_annotation*
- Download W2V2-LL4300h [here](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA/tree/main/LL_4300)
- Download W2V2-Pro or W2V2-MyST [here](https://huggingface.co/lijialudew/wav2vec_Providence/tree/main)
- **Copy *wav2vec2.ckpt* from the W2V2-Pro checkpoint folder to the <save_folder> path specified in the *hparams* file (if <save_folder> does not exists, create one)**

### Fine-tune wav2vec2 model on speaker diarization and parent/infant vocalization classification tasks ###
Before running Python script, first run
```
cd recipes/BabbleCor
```

Run the following commands to fine-tune wav2vec2 using our developed recipe

```
# Fine-tune wav2vec2-LL4300h only
python scripts/train_1_w2v2_WA_2dnn.py hparams/train_1_w2v2_2dnn_WA_LL4300_bbcor.yaml

# Train wav2vec2 with ASR features 
python scripts/train_1_w2v2_WA_2dnn_combine_asr_features_bbcor.py hparams/train_1_w2v2_2dnn_WA_LL4300_asr_bbcor_concat.yaml
```

### Paper/BibTex Citation
If you found this recipe or our paper helpful, please cite us as

```
Coming soon
```

### Contact
Jialu Li (she, her, hers)

Ph.D candidate @ Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign

E-mail: jialuli3@illinois.edu

Homepage: https://sites.google.com/view/jialuli/