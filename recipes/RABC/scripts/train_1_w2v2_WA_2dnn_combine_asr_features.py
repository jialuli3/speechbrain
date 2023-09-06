#!/usr/bin/env python3

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import numpy as np

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        
        batch = batch.to(self.device)
        wavs_adu, lens_adu = batch.sig_adu
        wavs_chi, lens_chi = batch.sig_chi

        outputs_adu = self.modules.wav2vec2(wavs_adu)
        outputs_chi = self.modules.wav2vec2(wavs_chi)
        # last dim will be used for AdaptativeAVG pool

        if self.hparams.combine == "sum":
            asr_features = self.get_asr_features(wavs_chi)
            outputs_adu = self.get_wa_outputs(outputs_adu,self.modules.weighted_average_adu, mean_pool_first=self.hparams.mean_pool_first_adu, lens = lens_adu)
            outputs_chi = self.get_wa_outputs(outputs_chi,self.modules.weighted_average_chi, mean_pool_first=self.hparams.mean_pool_first_chi, lens = lens_chi)
            outputs_chi = (1-self.hparams.combine_factor_asr)*outputs_chi + self.hparams.combine_factor_asr * asr_features

            outputs_adu_comb = outputs_adu + self.hparams.combine_factor * outputs_chi
            outputs_chi_comb = outputs_chi + self.hparams.combine_factor * outputs_adu
        elif self.hparams.combine == "concat":
            asr_features = self.get_asr_features(wavs_chi)
            outputs_adu = self.get_wa_outputs(outputs_adu,self.modules.weighted_average_adu, mean_pool_first=self.hparams.mean_pool_first_adu, lens = lens_adu)
            outputs_chi = self.get_wa_outputs(outputs_chi,self.modules.weighted_average_chi, mean_pool_first=self.hparams.mean_pool_first_chi, lens = lens_chi)
            if self.hparams.combine_asr == "concat":
                outputs_chi = torch.cat((outputs_chi, asr_features), dim=1)
            else:
                outputs_chi = (1-self.hparams.combine_factor_asr)*outputs_chi + self.hparams.combine_factor_asr * asr_features

            outputs_adu_comb = torch.cat((outputs_adu,outputs_chi), dim =1)
            outputs_chi_comb = torch.cat((outputs_chi,outputs_adu), dim =1)
        else: # cross-att (an experimental condition not included in original paper), reduce time dimension to not mean pool first
            asr_features = self.get_asr_features(wavs_chi)
            #asr_features = self.get_asr_features(wavs_chi, apply_avg=False)
            outputs_adu = outputs_adu.permute(1,2,3,0)
            outputs_adu = self.modules.weighted_average_adu(outputs_adu) #B,T,D
            outputs_chi = outputs_chi.permute(1,2,3,0)
            outputs_chi = self.modules.weighted_average_chi(outputs_chi) #B,T,D
            #outputs_chi = (1-self.hparams.combine_factor_asr)*outputs_chi + self.hparams.combine_factor_asr * asr_features

            outputs_adu_comb, _ = self.modules.multihead_att_adu(outputs_adu, outputs_chi, outputs_chi)
            outputs_chi_comb, _ = self.modules.multihead_att_chi(outputs_chi, outputs_adu, outputs_adu)

            outputs_adu_comb = self.hparams.avg_pool(outputs_adu_comb,lens_adu)
            outputs_adu_comb = outputs_adu_comb.view(outputs_adu_comb.shape[0],-1)
            outputs_chi_comb = self.hparams.avg_pool(outputs_chi_comb,lens_chi)
            outputs_chi_comb = outputs_chi_comb.view(outputs_chi_comb.shape[0],-1)
            #utt-level fusion
            if self.hparams.combine_asr == "concat":
                outputs_chi_comb = torch.cat((outputs_chi_comb, asr_features), dim=1)
            else:
                outputs_chi_comb = (1-self.hparams.combine_factor_asr)*outputs_chi_comb + self.hparams.combine_factor_asr * asr_features

        outputs_adu_comb = self.modules.output_mlp_adu(self.modules.dnn_adu(outputs_adu_comb))
        outputs_chi_comb = self.modules.output_mlp_chi(self.modules.dnn_chi(outputs_chi_comb))

        return outputs_adu_comb, outputs_chi_comb
    
    def get_asr_features(self, wavs, apply_avg=True):
        with torch.no_grad():
            embeddings = self.hparams.wav2vec_asr(wavs)
            if apply_avg:
                embeddings = self.hparams.avg_pool(embeddings).squeeze_()
        return embeddings
    
    def get_wa_outputs(self, outputs, wa, mean_pool_first=True, lens=None):
        if mean_pool_first:
            avg_outputs = []
            for i in range(len(outputs)):
                avg_output = self.hparams.avg_pool(outputs[i],lens)
                avg_output = avg_output.view(avg_output.shape[0],-1)
                avg_outputs.append(avg_output)
            outputs = torch.stack(avg_outputs).permute(1,2,0)
            outputs = wa(outputs)
        else: #WA first
            outputs = outputs.permute(1,2,3,0)
            outputs = wa(outputs) #B,T,D
            outputs = self.hparams.avg_pool(outputs,lens)
            outputs = outputs.view(outputs.shape[0],-1)
    
        return outputs

    def compute_objectives(self, outputs_adu, outputs_chi, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        adu_true, chi_true = batch.adu_true, batch.chi_true

        """to meet the input form of nll loss"""
        predictions_adu = self.hparams.log_softmax(outputs_adu)
        predictions_chi = self.hparams.log_softmax(outputs_chi)

        loss = 0.5 * self.hparams.compute_cost(predictions_adu, adu_true) + 0.5 * self.hparams.compute_cost(predictions_chi, chi_true)
    
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions_adu, adu_true, predictions_chi, chi_true)
        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions_adu, predictions_chi = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions_adu, predictions_chi, batch, sb.Stage.TRAIN)
        
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self,batch,stage):
        predictions_adu, predictions_chi = self.compute_forward(batch, stage)
        loss = self.compute_objectives(predictions_adu, predictions_chi, batch, stage)
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
            self.error_metrics = self.hparams.error_stats()

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
                "error_rate_f1": 1-(0.5*self.error_metrics.summarize("macro_f1_adu")+\
                    0.5*self.error_metrics.summarize("macro_f1_chi"))
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate_f1"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate_f1"])
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
                min_keys=["error_rate_f1"]
            )

            with open(self.hparams.train_log, "a") as w:
                self.error_metrics.write_stats(w)
        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            with open(self.hparams.train_log, "a") as w:
                self.error_metrics.write_stats(w)

            if self.hparams.output_annotation:
                if not os.path.exists(self.hparams.out_annotation):
                    os.mkdir(self.hparams.out_annotation)
                self.error_metrics.write_out_labels(os.path.join(self.hparams.out_annotation,os.path.basename(self.hparams.test_annotation)))
                
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
    @sb.utils.data_pipeline.takes("wav_adu")
    @sb.utils.data_pipeline.provides("sig_adu")
    def audio_pipeline_adu(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        if len(sig.shape)>=2:
            sig = sig[:, 0]
        return sig

    @sb.utils.data_pipeline.takes("wav_chi")
    @sb.utils.data_pipeline.provides("sig_chi")
    def audio_pipeline_chi(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        if len(sig.shape)>=2:
            sig = sig[:, 0]
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("ADU")
    @sb.utils.data_pipeline.provides("adu_true")
    def label_pipeline_adu(input):
        dict_map={"vocalization":1,"laughter":2,"N":0}
        if input in dict_map:
            yield dict_map[input]
        yield 0

    #chi, fan, and man default value as 0 for silence
    @sb.utils.data_pipeline.takes("CHI")
    @sb.utils.data_pipeline.provides("chi_true")
    def label_pipeline_chi(input):
        dict_map={"vocalization":1,"verbalization":2,"laugh":3,"whine_cry":4,"N":0}
        if input in dict_map:
            yield dict_map[input]
        yield 0

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline_adu, audio_pipeline_chi, label_pipeline_adu, label_pipeline_chi],
            output_keys=["id", "sig_adu", "sig_chi", "adu_true", "chi_true"],
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

    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    hparams["pretrainer"].load_collected(device=run_opts["device"])
    hparams["wav2vec_asr"].eval()
    hparams["wav2vec_asr"] = hparams["wav2vec_asr"].to(run_opts["device"])

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
    if hparams["output_annotation"] and not os.path.exists(hparams["out_annotation"]):
        os.mkdir(hparams["out_annotation"])

    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate_f1",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
