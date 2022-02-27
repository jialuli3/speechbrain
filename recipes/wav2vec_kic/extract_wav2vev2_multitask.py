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
    hparams["checkpointer"].recover_if_possible()
    
    #for dataset_type in ["train","valid","test"]:
        #json_data=load_json(hparams["{}_annotation".format(dataset_type)])
        # extract w2v2 output features
        # for key,entry in json_data.items():
        #     wav = entry["wav"]
        #     len = torch.FloatTensor([entry["dur"]])
        #     sig = sb.dataio.dataio.read_audio(wav).unsqueeze(0)
        #     sig, len =sig.to(device="cuda:0"), len.to(device="cuda:0")
        #     outputs = hparams["wav2vec2"](sig)

        #     # last dim will be used for AdaptativeAVG pool
        #     outputs = hparams["avg_pool"](outputs, len)
        #     outputs = outputs.view(outputs.shape[0], -1).squeeze()
        #     outputs = outputs.detach().cpu().numpy()
        #     entry["w2v2"]=os.path.join(hparams["w2v2_feature_out_root"],"dev",key)
        #     sb.dataio.dataio.save_pkl(outputs,entry["w2v2"])

        # extract w2v2 output convolution features
    all_json_files=sorted(glob.glob(hparams["train_annotations"]))
    for curr_json_file in all_json_files: 
        print(curr_json_file)
        if os.path.exists(os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file))):
            continue
        json_data=load_json(curr_json_file)
        #datasets = dataio_prep(hparams,curr_json_file)
        #target_dataloader = sb.dataio.dataloader.make_dataloader(datasets["train"],shuffle=False,batch_size=hparams["batch_size"])

        all_w2v2_outputs=[]
        # for i, batch in enumerate(target_dataloader):
        #     batch = batch.to("cuda:0")
        #     wavs, lens = batch.sig
        #     outputs = hparams["wav2vec2"](wavs)

        #     outputs = hparams["avg_pool"](outputs, lens)
        #     outputs = outputs.view(outputs.shape[0], -1).squeeze()
        #     outputs = outputs.detach().cpu().numpy()
        #     all_w2v2_outputs.extend(outputs)
        # for i,(key,entry) in enumerate(json_data.items()):
        #     entry["w2v2_conv"]=os.path.join(hparams["w2v2_conv_feature_out_root"],key)
        #     sb.dataio.dataio.save_pkl(all_w2v2_outputs[i],entry["w2v2_conv"])

        for key,entry in json_data.items():
            print(key)
            wav = entry["wav"]
            #len = torch.FloatTensor([entry["dur"]])
            len = torch.FloatTensor([2.0])
            sig = sb.dataio.dataio.read_audio(wav).unsqueeze(0)
            sig, len =sig.to(device="cuda:0"), len.to(device="cuda:0")
            outputs = hparams["wav2vec2"](sig)

            # last dim will be used for AdaptativeAVG pool
            outputs = hparams["avg_pool"](outputs, len)
            outputs = outputs.view(outputs.shape[0], -1).squeeze()
            outputs = outputs.detach().cpu().numpy()
            entry["w2v2_conv"]=os.path.join(hparams["w2v2_conv_feature_out_root"],key)
            sb.dataio.dataio.save_pkl(outputs,entry["w2v2_conv"])

        write_json(json_data,os.path.join(hparams["out_json_prefix"],os.path.basename(curr_json_file)))  
