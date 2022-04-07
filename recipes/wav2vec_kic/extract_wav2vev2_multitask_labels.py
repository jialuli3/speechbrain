#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP

Authors
 * Yingzhi WANG 2021
"""

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import json
import numpy as np
import glob

def dataio_prep(hparams,curr_json):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=curr_json,
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig"],
        )

    return datasets

def write_json(content,out):
    f=open(out,"w")
    f.write(json.dumps(content, sort_keys=False, indent=2))
    f.close()

def load_json(path):
    f=open(path,"r")
    return json.load(f)

def outputs2labels(predictions_sp,predictions_chn,predictions_fan,predictions_man):
    dict_map_sp={1:"CHN",2:"FAN",3:"MAN",4:"CXN",0:"SIL"}    
    dict_map_chn={0:"CRY",1:"FUS",2:"BAB"}
    dict_map_fan={0:"CDS",1:"FAN",2:"LAU",3:"SNG"}
    dict_map_man={0:"CDS",1:"MAN",2:"LAU",3:"SNG"}
    
    labels_sp,labels_chn,labels_fan,labels_man=[],[],[],[]

    for i,pred in enumerate(predictions_sp):
        if dict_map_sp[pred]=="SIL":
            labels_sp.append("SIL")
            labels_chn.append("SIL")
            labels_fan.append("SIL")
            labels_man.append("SIL")
        elif dict_map_sp[pred]=="CHN":
            labels_sp.append(dict_map_sp[pred])
            labels_chn.append(dict_map_chn[predictions_chn[i]])
            labels_fan.append("SIL")
            labels_man.append("SIL")        
        elif dict_map_sp[pred]=="FAN":
            labels_sp.append(dict_map_sp[pred])
            labels_chn.append("SIL")
            labels_fan.append(dict_map_fan[predictions_fan[i]])
            labels_man.append("SIL") 
        elif dict_map_sp[pred]=="MAN":
            labels_sp.append(dict_map_sp[pred])
            labels_chn.append("SIL")
            labels_fan.append("SIL")             
            labels_man.append(dict_map_man[predictions_man[i]])
        else:
            labels_sp.append(dict_map_sp[pred])
            labels_chn.append("SIL")
            labels_fan.append("SIL")
            labels_man.append("SIL")
    return labels_sp,labels_chn,labels_fan,labels_man    

# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    #sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    hparams["wav2vec2"] = hparams["wav2vec2"].to("cuda:0")
    hparams["output_mlp_sp"] = hparams["output_mlp_sp"].to("cuda:0")
    hparams["output_mlp_chn"] = hparams["output_mlp_chn"].to("cuda:0")
    hparams["output_mlp_fan"] = hparams["output_mlp_fan"].to("cuda:0")
    hparams["output_mlp_man"] = hparams["output_mlp_man"].to("cuda:0")

    hparams["checkpointer"].recover_if_possible()

    all_json_files=sorted(glob.glob(hparams["train_annotations"]))
    for curr_json_file in all_json_files:
        print(curr_json_file)
        # if os.path.exists(os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file))):
        #     continue        
        datasets = dataio_prep(hparams,curr_json_file)
        target_dataloader = sb.dataio.dataloader.make_dataloader(datasets["train"],shuffle=False,batch_size=hparams["batch_size"])

        labels_sp,labels_chn,labels_fan,labels_man=[],[],[],[]
        all_w2v2_outputs=[]
        for i, batch in enumerate(target_dataloader):
            batch = batch.to("cuda:0")
            wavs, lens = batch.sig
            outputs_conv = hparams["wav2vec2"].extract_features(wavs)
            outputs = hparams["wav2vec2"](wavs)

            # last dim will be used for AdaptativeAVG pool
            outputs = hparams["avg_pool"](outputs, lens)
            outputs = outputs.view(outputs.shape[0], -1)
            
            outputs_sp = hparams["output_mlp_sp"](outputs)
            outputs_chn = hparams["output_mlp_chn"](outputs)
            outputs_fan = hparams["output_mlp_fan"](outputs)    
            outputs_man = hparams["output_mlp_man"](outputs)

            predictions_sp = hparams["log_softmax"](outputs_sp)
            predictions_chn = hparams["log_softmax"](outputs_chn)
            predictions_fan = hparams["log_softmax"](outputs_fan)
            predictions_man = hparams["log_softmax"](outputs_man)

            predictions_sp = np.argmax(predictions_sp.cpu().detach().numpy(),axis=1)
            predictions_chn = np.argmax(predictions_chn.cpu().detach().numpy(),axis=1)
            predictions_fan = np.argmax(predictions_fan.cpu().detach().numpy(),axis=1)
            predictions_man = np.argmax(predictions_man.cpu().detach().numpy(),axis=1)

            outputs = outputs.view(outputs.shape[0], -1).squeeze()
            outputs = outputs.detach().cpu().numpy()
            all_w2v2_outputs.extend(outputs)

            curr_labels_sp,curr_labels_chn,curr_labels_fan,curr_labels_man=outputs2labels(predictions_sp,predictions_chn,predictions_fan,predictions_man)
            # assert(len(curr_labels_sp)==hparams["batch_size"])
            # assert(len(curr_labels_chn)==hparams["batch_size"])
            # assert(len(curr_labels_fan)==hparams["batch_size"])
            # assert(len(curr_labels_man)==hparams["batch_size"])

            labels_sp.extend(curr_labels_sp)
            labels_chn.extend(curr_labels_chn)
            labels_fan.extend(curr_labels_fan)
            labels_man.extend(curr_labels_man)

        json_data=load_json(curr_json_file)
        for i,(key,entry) in enumerate(json_data.items()):
            #if i>=len(labels_sp): continue
            entry["sp"]=labels_sp[i]
            entry["chn"]=labels_chn[i]
            entry["fan"]=labels_fan[i]
            entry["man"]=labels_man[i]
            entry["w2v2_conv"]=os.path.join(hparams["w2v2_conv_feature_out_root"],key)
            sb.dataio.dataio.save_pkl(all_w2v2_outputs[i],entry["w2v2_conv"])
        write_json(json_data,os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file)))
