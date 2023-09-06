"""The ``metric_stats`` module provides an abstract class for storing
statistics produced over the course of an experiment and summarizing them.

Authors:
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
"""
from cgitb import text
from concurrent.futures.process import _chain_from_iterable_of_lists
import enum
from lzma import PRESET_DEFAULT
from turtle import pos
import torch
from joblib import Parallel, delayed
from praatio import textgrid
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch
from speechbrain.dataio.dataio import merge_char, split_word
from speechbrain.dataio.wer import print_wer_summary, print_alignments

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,cohen_kappa_score,recall_score
import numpy as np
import pdb
import json
import os


def write_json(content,out):
    f=open(out,"w")
    f.write(json.dumps(content, sort_keys=False, indent=2))
    f.close()

def load_json(path):
    f=open(path,"r")
    return json.load(f)

class MetricStats:
    """A default class for storing and summarizing arbitrary metrics.

    More complex metrics can be created by sub-classing this class.

    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metric. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.
        Not usually used in sub-classes.
    batch_eval: bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.

    Example
    -------
    >>> from speechbrain.nnet.losses import l1_loss
    >>> loss_stats = MetricStats(metric=l1_loss)
    >>> loss_stats.append(
    ...      ids=["utterance1", "utterance2"],
    ...      predictions=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    ...      targets=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    ...      reduction="batch",
    ... )
    >>> stats = loss_stats.summarize()
    >>> stats['average']
    0.050...
    >>> stats['max_score']
    0.100...
    >>> stats['max_id']
    'utterance2'
    """

    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args, **kwargs
            Arguments to pass to the metric function.
        """
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.metric(*args, **kwargs).detach()

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores = sequence_evaluation(metric=self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """
        
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "average": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()

        message = f"Average score: {self.summary['average']}\n"
        message += f"Min error: {self.summary['min_score']} "
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']} "
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


def multiprocess_evaluation(metric, predict, target, lengths=None, n_jobs=8):
    """Runs metric evaluation if parallel over multiple jobs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    while True:
        try:
            scores = Parallel(n_jobs=n_jobs, timeout=30)(
                delayed(metric)(p, t) for p, t in zip(predict, target)
            )
            break
        except Exception as e:
            print(e)
            print("Evaluation timeout...... (will try again)")

    return scores


def sequence_evaluation(metric, predict, target, lengths=None):
    """Runs metric evaluation sequentially over the inputs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    scores = []
    for p, t in zip(predict, target):
        score = metric(p, t)
        scores.append(score)
    return scores


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.

    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = cer_stats.summarize()
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        scores = wer_details_for_batch(ids, target, predict, True)

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)     

class ConfidenceScoreStats(MetricStats):
    """A class for tracking confidence scores without reference transcripts (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.

    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = cer_stats.summarize()
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token
        self.preds_raw = []
        self.confidence_scores = []
        self.preds_token = []
        self.durs = []

    def append(
        self,
        ids,
        logits,
        predict_token,
        dur,
        ind2lab,
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        dur : torch.tensor
            The duration of audio.

        """
        self.ids.extend(ids)
        max_word_ids=logits.argmax(-1).flatten().detach().cpu().numpy()
        confidence_scores = logits.amax(-1).flatten().detach().cpu().numpy()
        self.confidence_scores.append(confidence_scores)
        self.preds_raw.append([ind2lab[max_word_ids[i]] for i in range(len(max_word_ids))]) #raw unmerged character
        self.preds_token.append(predict_token) #raw unmerged character
        self.durs.append(dur)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        #possibly need to add character probability
        self.summary=[]
        self.chars=[]
        for i in range(len(self.preds_token)):
            self.summary.append(self.ids[i])
            pred_token=self.preds_token[i]
            self.summary.append(" ".join(pred_token[0]).lower())
            curr_dur=self.durs[i].detach().cpu().numpy()[0]
            curr_chars=[]
            for t,char in enumerate(self.preds_raw[i]):
                if str(char)!="<blank>":
                    self.summary.append(" ".join([str(round(t/len(self.preds_raw[i])*curr_dur,3)), str(round((t+1)/len(self.preds_raw[i])*curr_dur,3)), \
                        str(round(self.confidence_scores[i][t],5)), str(char).lower()]))
                curr_chars.append([round(t/len(self.preds_raw[i])*curr_dur,3), round((t+1)/len(self.preds_raw[i])*curr_dur,3), \
                        str(round(self.confidence_scores[i][t],5)), str(char).lower()])
            self.chars.append(curr_chars)
            self.summary.append("\n")

    def greedy_decode_helper(self,chars):
        new_chars=[chars[0]]
        #first remove duplicate chars, then remove blank
        for i in range(1,len(chars)):
            if chars[i]!=new_chars[-1]:
                new_chars.append(chars[i])
        new_chars=[x for x in new_chars if x!="<blank>"]
        return new_chars

    def write_stats(self, filestream, tg_prefix):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        self.summarize()
        f=open(filestream,"w")
        f.writelines("\n".join(self.summary))
        f.close()

        for i in range(len(self.ids)):
            tg = textgrid.Textgrid()
            word_interval=[]
            curr_chars=[]
            start,end=0,0
            for curr_start,curr_end,_,curr_char in self.chars[i]:
                if curr_char==" ":
                    if len(curr_chars)>0:
                        curr_chars=self.greedy_decode_helper(curr_chars)
                        word_interval.append((start,end,"".join(curr_chars)))
                        curr_chars=[]
                        start=curr_end
                elif curr_char=="<blank>":
                    if len(curr_chars)>0:
                        curr_chars.append(curr_char)
                        #end=curr_end
                else:
                    if len(curr_chars)==0:
                        start=curr_start
                    curr_chars.append(curr_char)
                    end=curr_end
            if len(curr_chars)>0:
                curr_chars=self.greedy_decode_helper(curr_chars)
                word_interval.append((start,end,"".join(curr_chars)))
            new_tier=textgrid.IntervalTier("ASR",word_interval,0,60)
            tg.addTier(new_tier)
            textgrid_file = tg_prefix+"/"+self.ids[i].replace(".wav",".TextGrid")
            tg.save(textgrid_file,format="long_textgrid",includeBlankSpaces=False)

class InferenceOutStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.

    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)
        self.scores.extend(predict)

    def write_stats(self, in_json_file, out_json_prefix):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        
        input_dict = load_json(in_json_file)
        for i,curr_id in enumerate(self.ids):
            input_dict[curr_id]["phonemes"]=self.scores[i]
        write_json(input_dict,os.path.join(out_json_prefix,os.path.basename(in_json_file)))   

class BinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self, positive_label=1):
        self.clear()
        self.positive_label = positive_label

    def clear(self):
        self.ids = []
        self.scores = []
        self.labels = []
        self.summary = {}

    def append(self, ids, scores, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples

        """
        self.ids.extend(ids)
        self.scores.extend(scores.detach())
        self.labels.extend(labels.detach())

    def summarize(self, field=None, threshold=None, beta=1, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - TP - True Positive
         - TN - True Negative
         - FP - False Positive
         - FN - False Negative
         - FAR - False Acceptance Rate
         - FRR - False Rejection Rate
         - DER - Detection Error Rate (EER if no threshold passed)
         - precision - Precision (positive predictive value)
         - recall - Recall (sensitivity)
         - F-score - Balance of precision and recall (equal if beta=1)
         - MCC - Matthews Correlation Coefficient

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.scores, list):
            self.scores = torch.stack(self.scores)
            self.labels = torch.stack(self.labels)

        if threshold is None:
            positive_scores = self.scores[self.labels.nonzero(as_tuple=True)]
            negative_scores = self.scores[
                self.labels[self.labels == 0].nonzero(as_tuple=True)
            ]

            eer, threshold = EER(positive_scores, negative_scores)

        pred = (self.scores >= threshold).float()
        true = self.labels

        TP = self.summary["TP"] = float(pred.mul(true).sum())
        TN = self.summary["TN"] = float((1.0 - pred).mul(1.0 - true).sum())
        FP = self.summary["FP"] = float(pred.mul(1.0 - true).sum())
        FN = self.summary["FN"] = float((1.0 - pred).mul(true).sum())

        self.summary["FAR"] = FP / (FP + TN + eps)
        self.summary["FRR"] = FN / (TP + FN + eps)
        self.summary["DER"] = (FP + FN) / (TP + TN + eps)

        self.summary["precision"] = TP / (TP + FP + eps)
        self.summary["recall"] = TP / (TP + FN + eps)
        self.summary["F-score"] = (
            (1.0 + beta ** 2.0)
            * TP
            / ((1.0 + beta ** 2.0) * TP + beta ** 2.0 * FN + FP)
        )

        self.summary["MCC"] = (TP * TN - FP * FN) / (
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
        ) ** 0.5

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

class RABCBinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.ids = []
        self.preds = []
        self.labels = []
        self.summary = {}

    def append(self, ids, preds, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples
        preds: dict
            list of predicted output
        labels: dict
            list of labels
        """
        self.ids.extend(ids)
        self.preds.extend(preds.detach())
        self.labels.extend(labels.detach())

    def summarize(self, field=None, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - accuracy 
         - weighted F1 score
         - macro F1 score
         - kappa scores
         - confusion matrices

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.preds, list):
            self.preds = np.argmax(torch.stack(self.preds).cpu().numpy(),axis=1)
            self.labels = torch.stack(self.labels).cpu().numpy()

        self.summarize_helper(self.preds,self.labels)
        
        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def summarize_helper(self, preds, labels):
        self.summary["accuracy"]=round(accuracy_score(labels,preds),3)
        self.summary["UAR"]=round(recall_score(labels,preds,average="macro"),3)
        self.summary["macro_f1"]=round(f1_score(labels,preds,average="macro"),3)
        self.summary["confusion_matrix"]=confusion_matrix(labels,preds)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()
        
        message = f"Accuracy: {self.summary['accuracy']}\n"
        message += f"UAR: {self.summary['UAR']}\n"
        message += f"macro f1: {self.summary['macro_f1']}\n"

        try:
            message += f"kappa_NOI: {self.summary['kappa_5']}\n"
        except:
            pass
        message += f"SIL Non-Can Can LAU CRY\n"
        message += f"{self.summary['confusion_matrix']}\n"

        filestream.write(message)
        if verbose:
            print(message)

class RABCMultiMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.ids = []
        self.preds_adu = []
        self.labels_adu = []
        self.preds_chi = []
        self.labels_chi = []
        self.summary = {}

    def append(self, ids, preds_adu, labels_adu, preds_chi, labels_chi):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples
        preds: dict
            list of predicted output
        labels: dict
            list of labels
        """
        self.ids.extend(ids)
        self.preds_adu.extend(preds_adu.detach())
        self.labels_adu.extend(labels_adu.detach())
        self.preds_chi.extend(preds_chi.detach())
        self.labels_chi.extend(labels_chi.detach())

    def summarize(self, field=None, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - accuracy 
         - weighted F1 score
         - macro F1 score
         - kappa scores
         - confusion matrices

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.preds_adu, list):
            self.preds_adu = np.argmax(torch.stack(self.preds_adu).cpu().numpy(),axis=1)
            self.labels_adu = torch.stack(self.labels_adu).cpu().numpy()
            self.preds_chi = np.argmax(torch.stack(self.preds_chi).cpu().numpy(),axis=1)
            self.labels_chi = torch.stack(self.labels_chi).cpu().numpy()

        self.summarize_helper(self.preds_adu,self.labels_adu, "adu")
        self.summarize_helper(self.preds_chi,self.labels_chi, "chi")
        
        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def summarize_helper(self, preds, labels, post_fix):
        self.summary["acc_{}".format(post_fix)]=round(accuracy_score(labels,preds),3)
        self.summary["UAR_{}".format(post_fix)]=round(recall_score(labels,preds,average="macro"),3)
        self.summary["macro_f1_{}".format(post_fix)]=round(f1_score(labels,preds,average="macro"),3)
        self.summary["confusion_matrix_{}".format(post_fix)]=confusion_matrix(labels,preds)

        # unique_labels=np.unique(labels)
        # for l in unique_labels:
        #     curr_ytrue=np.where(labels==l,1,0)
        #     curr_ypred=np.where(preds==l,1,0)
        #     self.summary["macro_f1_{}_{}".format(str(l),post_fix)]=round(f1_score(curr_ytrue,curr_ypred,average="macro"),3)
        self.summary["macro_f1_overall_{}".format(post_fix)]=f1_score(labels,preds,average=None)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()
        
        message = f"Accuracy adu: {self.summary['acc_adu']}\n"
        message += f"macro f1 adu: {self.summary['macro_f1_adu']}\n"

        message += f"Confusion matrix:\n"
        message += f"SIL VOC LAU\n"
        message += f"{self.summary['confusion_matrix_adu']}\n"
        message += f"{self.summary['macro_f1_overall_adu']}\n"

        message += f"Accuracy chi: {self.summary['acc_chi']}\n"
        message += f"macro f1 chi: {self.summary['macro_f1_chi']}\n"

        message += f"Confusion matrix:\n"
        message += f"SIL VOC VERB LAU CRY\n"
        message += f"{self.summary['confusion_matrix_chi']}\n"
        message += f"{self.summary['macro_f1_overall_chi']}\n"
        # message += f"SIL f1 {self.summary['macro_f1_0_chi']}\n"
        # message += f"VOC f1 {self.summary['macro_f1_1_chi']}\n"
        # message += f"VERB f1 {self.summary['macro_f1_2_chi']}\n"
        # message += f"LAU f1 {self.summary['macro_f1_3_chi']}\n"
        # message += f"CRY f1 {self.summary['macro_f1_4_chi']}\n"

        filestream.write(message)
        if verbose:
            print(message)
        
    def write_out_labels(self, out_json_file):
        out_dict={}
        adu_dict_map={1:"vocalization",2:"laughter",0:"N"}
        chi_dict_map={1:"vocalization",2:"verbalization",3:"laugh",4:"whine_cry",0:"N"}

        for i,curr_id in enumerate(self.ids):
            out_dict[curr_id]={
                "ADU": adu_dict_map[self.labels_adu[i]],
                "CHI": chi_dict_map[self.labels_chi[i]],
                "ADU_pred": adu_dict_map[self.preds_adu[i]],
                "CHI_pred": chi_dict_map[self.preds_chi[i]],
            }
        write_json(out_dict, out_json_file)        

class RABCMultiMetricStatsAff(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.ids_sp = []
        self.preds_sp = []
        self.labels_sp = []
        self.ids_voc = []
        self.preds_voc = []
        self.labels_voc = []
        self.ids_aff = []
        self.preds_aff = []
        self.labels_aff = []
        self.summary = {}


    def append(self, ids_sp, preds_sp, labels_sp, ids_voc, preds_voc, labels_voc, ids_aff=None, preds_aff=None, labels_aff=None):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples
        preds: dict
            list of predicted output
        labels: dict
            list of labels
        """
        self.ids_sp.extend(ids_sp)
        self.preds_sp.extend(preds_sp.detach())
        self.labels_sp.extend(labels_sp.detach())
        if len(preds_voc)>0:
            self.ids_voc.extend(ids_voc)
            self.preds_voc.extend(preds_voc.detach())
            self.labels_voc.extend(labels_voc.detach())
        if len(preds_aff)>0:
            self.ids_aff.extend(ids_aff)
            self.preds_aff.extend(preds_aff.detach())
            self.labels_aff.extend(labels_aff.detach())            

    def summarize(self, field=None, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - accuracy 
         - weighted F1 score
         - macro F1 score
         - kappa scores
         - confusion matrices

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.preds_sp, list):
            self.preds_sp = np.argmax(torch.stack(self.preds_sp).cpu().numpy(),axis=1)
            self.labels_sp = torch.stack(self.labels_sp).cpu().numpy()
            self.preds_voc = np.argmax(torch.stack(self.preds_voc).cpu().numpy(),axis=1)
            self.labels_voc = torch.stack(self.labels_voc).cpu().numpy()
            self.preds_aff = np.argmax(torch.stack(self.preds_aff).cpu().numpy(),axis=1)
            self.labels_aff = torch.stack(self.labels_aff).cpu().numpy()

        self.summarize_helper(self.preds_sp,self.labels_sp, "sp")
        self.summarize_helper(self.preds_voc,self.labels_voc, "voc")
        self.summarize_helper(self.preds_aff,self.labels_aff, "aff")

        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def summarize_helper(self, preds, labels, post_fix="sp"):
        self.summary["acc_{}".format(post_fix)]=round(accuracy_score(labels,preds),3)
        self.summary["UAR_{}".format(post_fix)]=round(recall_score(labels,preds,average="macro"),3)
        self.summary["macro_f1_{}".format(post_fix)]=round(f1_score(labels,preds,average="macro"),3)
        self.summary["confusion_matrix_{}".format(post_fix)]=confusion_matrix(labels,preds)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()
        
        message = f"Accuracy sp: {self.summary['acc_sp']}\n"
        message += f"UAR sp: {self.summary['UAR_sp']}\n"
        message += f"macro f1 sp: {self.summary['macro_f1_sp']}\n"

        message += f"Confusion matrix:\n"
        message += f"N/A Speech\n"
        message += f"{self.summary['confusion_matrix_sp']}\n"

        message += f"Accuracy voc: {self.summary['acc_voc']}\n"
        message += f"UAR voc: {self.summary['UAR_voc']}\n"
        message += f"macro f1 voc: {self.summary['macro_f1_voc']}\n"

        message += f"Confusion matrix:\n"
        message += f"vocal verbal\n"
        message += f"{self.summary['confusion_matrix_voc']}\n"

        message += f"Accuracy aff: {self.summary['acc_aff']}\n"
        message += f"UAR aff: {self.summary['UAR_aff']}\n"
        message += f"macro f1 aff: {self.summary['macro_f1_aff']}\n"

        message += f"Confusion matrix:\n"
        message += f"cry laugh\n"
        message += f"{self.summary['confusion_matrix_aff']}\n"


        filestream.write(message)
        if verbose:
            print(message)

class KICBinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.ids = []
        self.preds = []
        self.labels = []
        # self.preds = {"sp":[],"chn":[],"fan":[],"man":[]}
        # self.labels = {"sp":[],"chn":[],"fan":[],"man":[]}
        self.summary = {}

    def append(self, ids, preds, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples
        preds: dict
            list of predicted output
        labels: dict
            list of labels
        """
        self.ids.extend(ids)
        self.preds.extend(preds.detach())
        self.labels.extend(labels.detach())

    def summarize(self, field=None, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - accuracy 
         - weighted F1 score
         - macro F1 score
         - kappa scores
         - confusion matrices

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.preds, list):
            self.preds = np.argmax(torch.stack(self.preds).cpu().numpy(),axis=1)
            self.labels = torch.stack(self.labels).cpu().numpy()

        self.summarize_helper(self.preds,self.labels,"")
        unique_labels=np.unique(self.labels)
        if len(unique_labels)>=6:
            ytrue_merge,ypred_merge=self.labels,self.preds
            ytrue_merge[ytrue_merge==5]=0
            ypred_merge[ypred_merge==5]=0
            self.summarize_helper(ypred_merge,ytrue_merge,"_merge")

        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def summarize_helper(self, preds, labels, merge, curr_type=""):
        self.summary["accuracy{}{}".format(curr_type,merge)]=round(accuracy_score(labels,preds),3)
        self.summary["macro_f1{}{}".format(curr_type,merge)]=round(f1_score(labels,preds,average="macro"),3)
        self.summary["kappa{}{}".format(curr_type,merge)]=round(cohen_kappa_score(labels,preds),3)

        unique_labels=np.unique(labels)
        for l in unique_labels:
            curr_ytrue=np.where(labels==l,1,0)
            curr_ypred=np.where(preds==l,1,0)
            self.summary["kappa_{}{}{}".format(str(l),curr_type,merge)]=round(cohen_kappa_score(curr_ytrue,curr_ypred),3)

        self.summary["confusion_matrix{}{}".format(curr_type,merge)]=confusion_matrix(labels,preds)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()
        
        message = f"Accuracy: {self.summary['accuracy']}\n"
        #message += f"weighted f1 score: {self.summary['weighted_f1']}\n"
        message += f"macro f1 score: {self.summary['macro_f1']}\n"
        message += f"kappa: {self.summary['kappa']}\n"
        # message += f"kappa_SIL: {self.summary['kappa_0']}\n"
        # message += f"kappa_CHN: {self.summary['kappa_1']}\n"
        # message += f"kappa_FAN: {self.summary['kappa_2']}\n"
        # message += f"kappa_MAN: {self.summary['kappa_3']}\n"
        # message += f"kappa_CXN: {self.summary['kappa_4']}\n"
        # try:
        #     message += f"kappa_NOI: {self.summary['kappa_5']}\n"
        # except:
        #     pass
        message += f"Confusion matrix:\n"
        message += f"SIL CHN  FAN  MAN  CXN \n"
        message += f"{self.summary['confusion_matrix']}\n"
        # if "kappa_merge" in self.summary.keys():
        #     message += f"Merge NOI and SIL into one category\n"
        #     message += f"Accuracy: {self.summary['accuracy_merge']}\n"
        #     message += f"weighted f1 score: {self.summary['weighted_f1_merge']}\n"
        #     message += f"macro f1 score: {self.summary['macro_f1_merge']}\n"
        #     message += f"kappa: {self.summary['kappa_merge']}\n"
        #     message += f"kappa_SIL: {self.summary['kappa_0_merge']}\n"
        #     message += f"kappa_CHN: {self.summary['kappa_1_merge']}\n"
        #     message += f"kappa_FAN: {self.summary['kappa_2_merge']}\n"
        #     message += f"kappa_MAN: {self.summary['kappa_3_merge']}\n"
        #     message += f"kappa_CXN: {self.summary['kappa_4_merge']}\n"
        #     message += f"Confusion matrix:\n"
        #     message += f"SIL CHN  FAN  MAN  CXN \n"
        #     message += f"{self.summary['confusion_matrix_merge']}\n"
        
        filestream.write(message)
        if verbose:
            print(message)


class KICMultitaskBinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """

    def __init__(self,record_man=True):
        self.clear()
        self.man=record_man

    def clear(self):
        self.ids = []
        self.preds = {"sp":[],"chn":[],"fan":[],"man":[]}
        self.labels = {"sp":[],"chn":[],"fan":[],"man":[]}
        self.summary = {}

    def append(self, ids, preds, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples
        preds: dict
            list of predicted output
        labels: dict
            list of labels
        """
        self.ids.extend(ids)
        self.preds["sp"].extend(preds[0].detach())
        self.preds["chn"].extend(preds[1].detach())
        self.preds["fan"].extend(preds[2].detach())
        if self.man:
            self.preds["man"].extend(preds[3].detach())

        self.labels["sp"].extend(labels[0].detach())
        self.labels["chn"].extend(labels[1].detach())
        self.labels["fan"].extend(labels[2].detach())
        if self.man:
            self.labels["man"].extend(labels[3].detach())

    def summarize(self, field=None, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - accuracy 
         - weighted F1 score
         - macro F1 score
         - kappa scores
         - confusion matrices

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        for tier in ["sp","chn","fan","man"]:
            if tier=="man" and (not self.man): continue
            self.curr_preds = np.argmax(torch.stack(self.preds[tier]).cpu().numpy(),axis=1)
            self.curr_labels = torch.stack(self.labels[tier]).cpu().numpy()
            
            self.summarize_helper(self.curr_preds,self.curr_labels,"",tier)
            if tier=="sp" and len(np.unique(self.curr_labels))>=6:
                ytrue_merge,ypred_merge=self.curr_labels,self.curr_preds
                ytrue_merge[ytrue_merge==5]=0
                ypred_merge[ypred_merge==5]=0
                self.summarize_helper(ypred_merge,ytrue_merge,"_merge","sp")

        if field is not None:
            return self.summary[field]
        else:
            return self.summary
    
    def summarize_helper(self, preds, labels, merge="", curr_type=""):
        self.summary["accuracy{}{}".format(curr_type,merge)]=round(accuracy_score(labels,preds),3)
        self.summary["weighted_f1{}{}".format(curr_type,merge)]=round(f1_score(labels,preds,average="weighted"),3)
        self.summary["macro_f1{}{}".format(curr_type,merge)]=round(f1_score(labels,preds,average="macro"),3)
        self.summary["kappa{}{}".format(curr_type,merge)]=round(cohen_kappa_score(labels,preds),3)

        unique_labels=np.unique(labels)
        for l in unique_labels:
            curr_ytrue=np.where(labels==l,1,0)
            curr_ypred=np.where(preds==l,1,0)
            self.summary["kappa_{}{}{}".format(str(l),curr_type,merge)]=round(cohen_kappa_score(curr_ytrue,curr_ypred),3)

        self.summary["confusion_matrix{}{}".format(curr_type,merge)]=confusion_matrix(labels,preds)

    def write_stats(self, filestream, verbose=True):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()

        message= f"Speaker Diarization\n"
        for tier in ["sp","chn","fan","man"]:
            if tier=="man" and (not self.man): continue
            message += f"Accuracy: {self.summary['accuracy{}'.format(tier)]}\n"
            #message += f"weighted f1 score: {self.summary['weighted_f1{}'.format(tier)]}\n"
            message += f"macro f1 score: {self.summary['macro_f1{}'.format(tier)]}\n"
            message += f"kappa: {self.summary['kappa{}'.format(tier)]}\n"
            # if tier=="sp":
            #     message += f"kappa_SIL: {self.summary['kappa_0{}'.format(tier)]}\n"
            #     message += f"kappa_CHN: {self.summary['kappa_1{}'.format(tier)]}\n"
            #     message += f"kappa_FAN: {self.summary['kappa_2{}'.format(tier)]}\n"
            #     if self.man:
            #         message += f"kappa_MAN: {self.summary['kappa_3{}'.format(tier)]}\n"
            #         message += f"kappa_CXN: {self.summary['kappa_4{}'.format(tier)]}\n"
            #     try:
            #         message += f"kappa_NOI: {self.summary['kappa_5{}'.format(tier)]}\n"
            #     except:
            #         pass
            # if tier=="chn":
            #     message += f"kappa_CRY: {self.summary['kappa_0{}'.format(tier)]}\n"
            #     message += f"kappa_FUS: {self.summary['kappa_1{}'.format(tier)]}\n"
            #     message += f"kappa_BAB: {self.summary['kappa_2{}'.format(tier)]}\n"

            # if tier=="fan" :
            #     message += f"kappa_CDS: {self.summary['kappa_0{}'.format(tier)]}\n"
            #     message += f"kappa_FAN: {self.summary['kappa_1{}'.format(tier)]}\n"
            #     message += f"kappa_LAU: {self.summary['kappa_2{}'.format(tier)]}\n"
            #     message += f"kappa_SNG: {self.summary['kappa_3{}'.format(tier)]}\n"

            # if tier=="man" and self.man:
            #     message += f"kappa_CDS: {self.summary['kappa_0{}'.format(tier)]}\n"
            #     message += f"kappa_MAN: {self.summary['kappa_1{}'.format(tier)]}\n"
            #     message += f"kappa_LAU: {self.summary['kappa_2{}'.format(tier)]}\n"
            #     message += f"kappa_SNG: {self.summary['kappa_3{}'.format(tier)]}\n"

            message += f"Confusion matrix:\n"
            if tier == "sp":
                message += f"SIL CHN  FAN  MAN  CXN \n"
            if tier =="chn":
                message += f"CRY FUS BAB \n"
            if tier =="fan":
                message += f"CDS  FAN  LAU  SNG \n"
            if tier =="man" and self.man:
                message += f"CDS  MAN  LAU  SNG \n"

            message += f"{self.summary['confusion_matrix{}'.format(tier)]}\n"

            if tier=="sp" and "kappasp_merge" in self.summary.keys():
                message += f"Merge NOI and SIL into one category\n"
                message += f"Accuracy: {self.summary['accuracy{}_merge'.format(tier)]}\n"
                message += f"weighted f1 score: {self.summary['weighted_f1{}_merge'.format(tier)]}\n"
                message += f"macro f1 score: {self.summary['macro_f1{}_merge'.format(tier)]}\n"
                message += f"kappa: {self.summary['kappa{}_merge'.format(tier)]}\n"
                message += f"kappa_SIL: {self.summary['kappa_0{}_merge'.format(tier)]}\n"
                message += f"kappa_CHN: {self.summary['kappa_1{}_merge'.format(tier)]}\n"
                message += f"kappa_FAN: {self.summary['kappa_2{}_merge'.format(tier)]}\n"
                if self.man:
                    message += f"kappa_MAN: {self.summary['kappa_3{}_merge'.format(tier)]}\n"
                    message += f"kappa_CXN: {self.summary['kappa_4{}_merge'.format(tier)]}\n"
                message += f"Confusion matrix:\n"
                message += f"SIL CHN  FAN  MAN  CXN \n"
                message += f"{self.summary['confusion_matrix{}_merge'.format(tier)]}\n"
        
        filestream.write(message)
        if verbose:
            print(message)

def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    # Finding the threshold for EER
    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])


def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).


    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])
