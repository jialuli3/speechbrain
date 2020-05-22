#!/usr/bin/python
import os
import torch
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average

torch.manual_seed(1234)
experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.abspath(experiment_dir + data_folder)
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


class ASR_Brain(sb.core.Brain):
    def compute_forward(self, x, train_mode=True, init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.linear1(feats, init_params)
        x = params.activation(x)
        x = params.linear2(x, init_params)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        predictions, lens = predictions
        ids, ali, ali_lens = targets
        loss = params.compute_cost(predictions, ali, [lens, ali_lens])

        if not train_mode:
            err = params.compute_error(predictions, ali, [lens, ali_lens])
            stats = {"error": err}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))


train_set = params.train_loader()
first_x, first_y = next(zip(*train_set))
asr_brain = ASR_Brain(
    modules=[params.linear1, params.linear2],
    optimizer=params.optimizer,
    first_inputs=[first_x],
)
train_stats, _ = asr_brain.fit(
    range(params.N_epochs), train_set, params.valid_loader()
)
test_stats = asr_brain.evaluate(params.test_loader())
print("Test error: %.2f" % summarize_average(test_stats["error"]))


# Define an integration test of overfitting on the train data
def test_error():
    assert summarize_average(train_stats["loss"]) < 0.2
