#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP

Authors
 * Yingzhi WANG 2021
"""

from hashlib import new
import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import json
import glob 
import numpy as np

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

def check_range(keys):
    #return keys.endswith("0.0") or keys.endswith("2.0") or keys.endswith("4.0") or keys.endswith("6.0") or keys.endswith("8.0")
    return keys.endswith(".0")


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    #datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to("cuda:0")
    if hparams["extract_labels"]:
        hparams["output_mlp_sp"] = hparams["output_mlp_sp"].to("cuda:0")
        hparams["output_mlp_chn"] = hparams["output_mlp_chn"].to("cuda:0")
        hparams["output_mlp_fan"] = hparams["output_mlp_fan"].to("cuda:0")
        hparams["output_mlp_man"] = hparams["output_mlp_man"].to("cuda:0")

    hparams["wav2vec2"].model.feature_extractor._freeze_parameters()
    hparams["checkpointer"].recover_if_possible(min_key="error_rate_kappa")

    # extract w2v2 output convolution features
    all_json_files=sorted(glob.glob(hparams["train_annotations"]))
    for curr_json_file in all_json_files: 
        print(curr_json_file)
        #if os.path.exists(os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file))): continue        
        json_data=load_json(curr_json_file)
        new_json_data={}
        for key,entry in json_data.items():
            start,end=key.split("_")[-2],key.split("_")[-1]
            print(key)
            #if not check_range(end): continue
            #prefix_dir="/".join(entry["wav"]["file"].split("/")[:-2])
            #wav = os.path.join(prefix_dir,"2s",key+".wav")

            wav=entry["wav"]
            try:
                sig = sb.dataio.dataio.read_audio(wav).unsqueeze(0)
            except:
                print(wav)
                continue
            len = torch.FloatTensor([entry["dur"]])
            #len = torch.FloatTensor([2.0])
            sig, len =sig.to(device="cuda:0"), len.to(device="cuda:0")
            outputs_conv = hparams["wav2vec2"].extract_features(sig)
            outputs_conv = hparams["avg_pool"](outputs_conv, len)
            outputs_conv = outputs_conv.detach().cpu().numpy().flatten()

            if hparams["extract_labels"]:
                outputs = hparams["wav2vec2"](sig)

                # last dim will be used for AdaptativeAVG pool
                outputs = hparams["avg_pool"](outputs, len)

                outputs_sp = hparams["output_mlp_sp"](outputs)
                outputs_chn = hparams["output_mlp_chn"](outputs)
                outputs_fan = hparams["output_mlp_fan"](outputs)    
                outputs_man = hparams["output_mlp_man"](outputs)

                prob_sp = hparams["softmax"](outputs_sp).flatten().cpu().detach().numpy()
                prob_chn = hparams["softmax"](outputs_chn).flatten().cpu().detach().numpy()
                prob_fan = hparams["softmax"](outputs_fan).flatten().cpu().detach().numpy()
                prob_man = hparams["softmax"](outputs_man).flatten().cpu().detach().numpy()

                predictions_sp = hparams["log_softmax"](outputs_sp).flatten()
                predictions_chn = hparams["log_softmax"](outputs_chn).flatten()
                predictions_fan = hparams["log_softmax"](outputs_fan).flatten()
                predictions_man = hparams["log_softmax"](outputs_man).flatten()

                predictions_sp = [np.argmax(predictions_sp.cpu().detach().numpy())]
                predictions_chn = [np.argmax(predictions_chn.cpu().detach().numpy())]
                predictions_fan = [np.argmax(predictions_fan.cpu().detach().numpy())]
                predictions_man = [np.argmax(predictions_man.cpu().detach().numpy())]

                out_sp,out_chn,out_fan,out_man=outputs2labels(predictions_sp,predictions_chn,predictions_fan,predictions_man)
                entry["sp"]=out_sp[0]
                entry["chn"]=out_chn[0]
                entry["fan"]=out_fan[0]
                entry["man"]=out_man[0]
                entry["sp_prob"]=str(max(prob_sp))
                entry["chn_prob"]=str(max(prob_chn))
                entry["fan_prob"]=str(max(prob_fan))
                entry["man_prob"]=str(max(prob_man))
            if hparams["compute_energy"]:
                sig = sig.cpu().detach().numpy().flatten()
                if max(sig)>1: sig/=32767
                entry["energy"]=str(max(-6,np.log10(np.mean(np.square(sig)))))
            if hparams["extract_w2v2_conv"]:
                if not os.path.exists(hparams["w2v2_conv_feature_out_root"]):
                    os.mkdir(hparams["w2v2_conv_feature_out_root"])
                entry["w2v2_conv"]=os.path.join(hparams["w2v2_conv_feature_out_root"],key)
                if not os.path.exists(entry["w2v2_conv"]):
                    sb.dataio.dataio.save_pkl(outputs_conv,entry["w2v2_conv"])
            new_json_data[key]=entry
        write_json(new_json_data,os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file)))  
        #write_json(new_json_data,hparams["out_json_prefix"])  
