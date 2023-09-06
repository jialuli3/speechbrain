## Rapid-ABC: clinician-child speaker diarizatio and vocalization classification analysis 
This recipe is developed based on SpeechBrain toolkit. This recipe contains scripts for training and testing child-adult speaker diarization and vocalization classifications using wav2vec 2.0 model on Rapid-ABC corpus.

## Uses
### Install SpeechBrain
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

### Check out this branch
```
git clone https://github.com/jialuli3/speechbrain.git
cd speechbrain
git checkout -b infant-voc-classification
git pull origin infant-voc-classification
```

### Prepare data in json format ###
To make data compatible with this script, prepare your data similar as the following json format (each entry corresponds to each frame)
```
{
  "sample_data1": { # silence interval
    "wav_adu": "path/to/your/wav/adult_wav",
    "wav_chi": "path/to/your/wav/child_wav",
    "dur": 2.0, # duration of each frame
    "ADU": "N",
    "CHI": "N",
    },
  "sample_data2": { # adult is talking, and child is laughing
    "wav_adu": "path/to/your/wav/adult_wav",
    "wav_chi": "path/to/your/wav/child_wav",
    "dur": 2.0, 
    "ADU": "vocalization",
    "CHI": "laugh",
    }
}
```
Vocalization types include:
- **ADU**
    - *vocalization*
    - *laugh*
    - *N*: silent/non-speech events
- **CHI**
    - *vocalization*: non-lexical vocalization
    - *verbalization*: contain words/phrases 
    - *laugh*
    - *cry*
    - *N*: silent/non-speech events


Sample json file we used in our experiments can be found in **sample_json/sample_json.json**

### Make yaml files in *hparams* folder compatiable with your dataset
- Change data paths in *train_annotation*, *valid_annotation*, and *test_annotation*
- Download W2V2-LL4300h [here](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA/tree/main/LL_4300)
- Download W2V2-Pro or W2V2-MyST [here](https://huggingface.co/lijialudew/wav2vec_Providence/tree/main)
- **Copy *wav2vec2.ckpt* from the W2V2-Pro checkpoint folder to the <save_folder> path specified in the *hparams* file (if <save_folder> does not exists, create one)**


### Fine-tune wav2vec2 model on speaker diarization and children vocalization classification tasks ###
Before running Python script, first run
```
cd recipes/RABC
```

Run the following commands to fine-tune wav2vec2 using our developed recipe

```
# Fine-tune W2V2-LL4300 only 
python scripts/train_1_w2v2_WA_2dnn.py hparams/train_1_w2v2_2dnn_WA_LL4300.yaml

# Fine-tune W2V2-LL4300 by combining both audio channels
python scripts/train_1_w2v2_WA_2dnn_combine.py hparams/train_1_w2v2_2dnn_WA_LL4300_concat.yaml

# Fine-tune W2V2-LL4300 by combining both audio channels and auxiliary ASR features
python scripts/train_1_w2v2_WA_2dnn_combine_asr_features.py hparams/train_1_w2v2_2dnn_WA_LL4300_concat_asr.yaml
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