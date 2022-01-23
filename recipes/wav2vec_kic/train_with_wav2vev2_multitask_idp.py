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

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs)

        # last dim will be used for AdaptativeAVG pool
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        outputs_sp = self.modules.output_mlp_sp(outputs)
        outputs_chn = self.modules.output_mlp_chn(outputs)
        outputs_fan = self.modules.output_mlp_fan(outputs)    

        return outputs_sp, outputs_chn, outputs_fan

    def compute_objectives(self, outputs_sp, outputs_chn, outputs_fan, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        sp_true, chn_true, fan_true = batch.sp_true, batch.chn_true, batch.fan_true
        """to meet the input form of nll loss"""
        predictions_sp = self.hparams.log_softmax(outputs_sp)
        predictions_chn = self.hparams.log_softmax(outputs_chn)
        predictions_fan = self.hparams.log_softmax(outputs_fan)

        predictions_chn=predictions_chn[(chn_true!=-1).nonzero(as_tuple=True)]
        predictions_fan=predictions_fan[(fan_true!=-1).nonzero(as_tuple=True)]

        chn_true=chn_true[(chn_true!=-1).nonzero(as_tuple=True)]
        fan_true=fan_true[(fan_true!=-1).nonzero(as_tuple=True)]

        loss = getattr(self.hparams,"sp_weights", 1) * self.hparams.compute_cost(predictions_sp, sp_true)
        if len(chn_true)!=0:
            loss+=getattr(self.hparams,"chn_weights", 1) * self.hparams.compute_cost(predictions_chn, chn_true)
        if len(fan_true)!=0:
            loss+=getattr(self.hparams,"fan_weights", 1) * self.hparams.compute_cost(predictions_fan, fan_true)
        
        if hasattr(self.hparams,"use_triplet_loss") and self.hparams.use_triplet_loss:
            def get_triplet_pos_neg_idx(targets_sp_vec,target_idx):
                anc_idx=((targets_sp_vec==target_idx)!=0).nonzero().flatten()
                pos_idx,neg_idx=[],[]
                if len(anc_idx)!=0:
                    pos_idx=anc_idx[torch.randint(anc_idx.size()[0],(anc_idx.size()[0],))]
                    neg_idx=((targets_sp_vec!=target_idx)!=0).nonzero().flatten()
                    neg_idx=neg_idx[torch.randint(neg_idx.size()[0],(anc_idx.size()[0],))]
                return anc_idx,pos_idx,neg_idx

            anc_idx,pos_idx,neg_idx=get_triplet_pos_neg_idx(sp_true,self.hparams.triplet_anchor_idx)
            if len(anc_idx)!=0:
                loss+=torch.maximum(torch.tensor(0).cuda(),\
                    self.hparams.triplet_loss_cost(outputs_sp[anc_idx,:],outputs_sp[pos_idx,:],outputs_sp[neg_idx,:])/len(anc_idx))
        
        if stage != sb.Stage.TRAIN:
            #self.error_metrics.append(batch.id, predictions_sp, sp_true)
            self.error_metrics_kic.append(batch.id,[predictions_sp, predictions_chn, predictions_fan],\
                [sp_true, chn_true, fan_true])
        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions_sp, predictions_chn, predictions_fan = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions_sp, predictions_chn, predictions_fan, batch, sb.Stage.TRAIN)
        
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self,batch,stage):
        predictions_sp, predictions_chn, predictions_fan = self.compute_forward(batch, stage)
        loss = self.compute_objectives(predictions_sp, predictions_chn, predictions_fan, batch, stage)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            #self.error_metrics = self.hparams.error_stats()
            self.error_metrics_kic = self.hparams.error_stats_kic()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                #"error_rate": self.error_metrics.summarize("average"),
                #"error_rate_kic": self.error_metrics_kic.summarize(),
                "error_rate_kappa": 1-(0.5*self.error_metrics_kic.summarize("kappasp")+
                0.3*self.error_metrics_kic.summarize("kappachn")+0.2*self.error_metrics_kic.summarize("kappafan"))
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate_kappa"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate_kappa"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, 
                min_keys=["error_rate_kappa"]
            )

            with open(self.hparams.train_log, "a") as w:
                self.error_metrics_kic.write_stats(w)
        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            with open(self.hparams.train_log, "a") as w:
                self.error_metrics_kic.write_stats(w)

            with open(self.hparams.output_log, "w") as w:
                self.error_metrics_kic.write_stats(w)
                
    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)


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
        if input.isnumeric():
            yield int(input)
        else:
            yield dict_map[input]
    
    #chn, fan, and man default value as 0 for silence
    @sb.utils.data_pipeline.takes("chn")
    @sb.utils.data_pipeline.provides("chn_true")
    def label_pipeline_chn(input):
        dict_map={"CRY":0,"FUS":1,"BAB":2,"N":-1,"LAU":-1,"SCR":-1}
        if input.isnumeric():
            yield int(input)-1
        else:
            yield dict_map[input]

    @sb.utils.data_pipeline.takes("fan")
    @sb.utils.data_pipeline.provides("fan_true")
    def label_pipeline_fan(input):
        dict_map={"CDS":0,"PLA":0,"PLAC": 0,"FAN":1,"LAU":2,"LAUC":2,"SNG":3,"SNGC":3,"SHH":-1,"N":-1}
        if input.isnumeric():
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
            dynamic_items=[audio_pipeline, label_pipeline_sp, label_pipeline_chn, label_pipeline_fan],
            output_keys=["id", "sig", "sp_true", "chn_true", "fan_true"],
        )

    return datasets


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

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate_kappa",
        test_loader_kwargs=hparams["dataloader_options"],
    )
