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

def dataio_prep(hparams):
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

    # Define opensmile pipeline
    # @sb.utils.data_pipeline.takes("os")
    # @sb.utils.data_pipeline.provides("os_value")
    # def os_pipeline(os_file):
    #     """Load the signal, and pass it and its length to the corruption class.
    #     This is done on the CPU in the `collate_fn`."""
    #     os_value = sb.dataio.dataio.load_pickle(os_file)
    #     return os_value

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("sp")
    @sb.utils.data_pipeline.provides("sp_true")
    def label_pipeline_sp(input):
        dict_map={"CHN":1,"FAN":2,"MAN":3,"CXN":4,"NOI":5,"SIL":0}
        if isinstance(input, int):
            yield int(input)
        else:
            yield dict_map[input]
    
    #chn, fan, and man default value as 0 for silence
    @sb.utils.data_pipeline.takes("chn")
    @sb.utils.data_pipeline.provides("chn_true")
    def label_pipeline_chn(input):
        dict_map={"CRY":0,"FUS":1,"BAB":2,"N":-1,"LAU":-1,"SCR":-1}
        if isinstance(input, int):
            yield int(input)-1
        else:
            yield dict_map[input]

    @sb.utils.data_pipeline.takes("fan")
    @sb.utils.data_pipeline.provides("fan_true")
    def label_pipeline_fan(input):
        dict_map={"CDS":0,"PLA":0,"PLAC": 0,"FAN":1,"LAU":2,"LAUC":2,"SNG":3,"SNGC":3,"SHH":-1,"N":-1}
        if isinstance(input, int):
            yield int(input)-1
        else:
            yield dict_map[input]

    @sb.utils.data_pipeline.takes("man")
    @sb.utils.data_pipeline.provides("man_true")
    def label_pipeline_man(input):
        dict_map={"CDS":0,"PLA":0,"PLAC": 0,"MAN":1,"LAU":2,"LAUC":2,"SNG":3,"SNGC":3,"SHH":-1,"N":-1}
        if isinstance(input, int):
            yield int(input)-1
        else:
            yield dict_map[input]

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline_sp, label_pipeline_chn, label_pipeline_fan, label_pipeline_man],
            output_keys=["id", "sig", "sp_true", "chn_true", "fan_true", "man_true"],
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
    datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to("cuda:0")
    # freeze the feature extractor part when unfreezing
    #if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
    #    hparams["wav2vec2"].model.feature_extractor._freeze_parameters()
    hparams["checkpointer"].recover_if_possible()
    
    json_data=load_json(hparams["train_annotation"])
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
    for key,entry in json_data.items():
        wav = entry["wav"]
        len = torch.FloatTensor([entry["dur"]])
        sig = sb.dataio.dataio.read_audio(wav).unsqueeze(0)
        sig, len =sig.to(device="cuda:0"), len.to(device="cuda:0")
        outputs = hparams["wav2vec2"].extract_features(sig)

        # last dim will be used for AdaptativeAVG pool
        outputs = hparams["avg_pool"](outputs, len)
        outputs = outputs.view(outputs.shape[0], -1).squeeze()
        outputs = outputs.detach().cpu().numpy()
        entry["w2v2_conv"]=os.path.join(hparams["w2v2_conv_feature_out_root"],"train",key)
        sb.dataio.dataio.save_pkl(outputs,entry["w2v2_conv"])
        
    write_json(json_data,hparams["train_annotation"])
  

    # for i,batch in enumerate(target_dataloader):

    #     batch = batch.to("cuda:0")
    #     wavs, lens = batch.sig
    #     print(wavs.size(),lens.size())
    #     outputs = hparams["wav2vec2"](wavs)

    #     # last dim will be used for AdaptativeAVG pool
    #     outputs = hparams["avg_pool"](outputs, lens)
    #     outputs = outputs.view(outputs.shape[0], -1)
        
    #     break

