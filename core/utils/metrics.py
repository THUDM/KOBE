import codecs
import logging
import os
import sys
import subprocess

import numpy as np
from sklearn import metrics

import pyrouge


def bleu(reference, candidate, log_path, print_log, config, lang="de", bpe=False):
    """
    return the bleu score, including multi-bleu and nist bleu.
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :param print_log: function to print log
    :param config: configuration
    :param lang: language
    :param bpe: use byte-pair encoding or not
    :return: bleu score
    """
    # check if there is specified reference file
    # if not, create one
    if config.refF != "":
        ref_file = config.refF
    else:
        ref_file = log_path+'reference.txt'
        with codecs.open(ref_file, 'w', 'utf-8') as f:
            for s in reference:
                if not config.char:
                    f.write(" ".join(s)+'\n')
                else:
                    f.write("".join(s) + '\n')
    cand_file = log_path+'candidate.txt'
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip()+'\n')
            else:
                f.write("".join(s).strip() + '\n')
    # file to store results
    temp = log_path + "result.txt"
    # implementation for German, nist bleu
    # if bpe:
    #     cand_wobpe = log_path+"cand_wobpe.txt"
    #     cmd_bpe_remove = "cat " + cand_file + \
    #         " | sed -E 's/(@@ )|(@@ ?$)//g' > " + cand_wobpe
    #     os.system(cmd_bpe_remove)
    # detok_ref_file = log_path+'detok_reference.txt'
    # detok_cand_file = log_path+'detok_candidate.txt'
    # cmd_detok_ref = "perl script/detokenize.perl -l " + \
    #     lang + " < " + ref_file + " > " + detok_ref_file
    # cmd_detok_cand = "perl script/detokenize.perl -l " + \
    #     lang + " < " + cand_wobpe + " > " + detok_cand_file
    # os.system(cmd_detok_ref)
    # os.system(cmd_detok_cand)
    # command = "perl script/multi-bleu-detok.perl " + \
    #     detok_ref_file + "<" + detok_cand_file + "> " + temp
    # run the multi-bleu perl script and get the score
    command = "perl core/utils/multi-bleu.perl -lc " + \
        ref_file + "<" + cand_file + "> " + temp

    try:
        subprocess.call(command, shell=True)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)

    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    # print the score
    print_log(result)

    return float(result.split()[2][:-1])


def rouge(reference, candidate, log_path, print_log, config):
    """
    compute the rouge score
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :param print_log: function to print log
    :param config: configuration
    :return: rouge-2 score
    """
    # check if of equal amount.
    assert len(reference) == len(candidate)
    # directory for saving sentences
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    # check if there are directories for reference and candidate
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    # write files
    for i in range(len(reference)):
        with codecs.open(ref_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(reference[i]).replace(' <\s> ', '\n') + '\n')
        with codecs.open(cand_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(candidate[i]).replace(
                ' <\s> ', '\n').replace('<unk>', 'UNK') + '\n')

    # use pyrouge and ROUGE155
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    # compute the scores
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    # recall
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    # precision
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    # f score
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    # print
    print_log("F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))

    return f_score[:], recall[:], precision[:]


def eval_metrics(reference, candidate, label_dict, log_path):
    """
    evaluation metric for multi-label classification
    :param reference: reference
    :param candidate: candidate
    :param label_dict: label dictionary
    :param log_path: path to log
    :return: hamming loss, macro and micro f1, precision and recall.
    """
    # directory for saving sentences
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    # check if there are directories for reference and candidate
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    ref_file = ref_dir+'reference'
    cand_file = cand_dir+'candidate'

    # write files
    for i in range(len(reference)):
        with codecs.open(ref_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(reference[i])+'\n')
        with codecs.open(cand_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(candidate[i])+'\n')

    def make_label(l, label_dict):
        """
        make labels with the dictionary
        :param l: candidate label set
        :param label_dict: label dictionary
        :return: one-hot representation of the label set
        """
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().upper(), 0) for label in l]
        result[indices] = 1
        return result[:]

    def prepare_label(y_list, y_pre_list, label_dict):
        """
        assemble the label sets into an array
        :param y_list: reference label sets
        :param y_pre_list: candidate label sets
        :param label_dict: label dictionary
        :return: one-hot arrays of the input label sets
        """
        reference = np.array([make_label(y, label_dict) for y in y_list])
        candidate = np.array([make_label(y_pre, label_dict)
                              for y_pre in y_pre_list])
        return reference, candidate

    def get_one_error(y, candidate, label_dict):
        """
        one error computation
        :param y: reference
        :param candidate: candidate
        :param label_dict: label dictionary
        :return: one error
        """
        idx = [label_dict.get(c[0].strip().upper(), 0) for c in candidate]
        result = [(y[i, idx[i]] == 1) for i in range(len(idx))]
        return (1 - np.array(result).mean())

    def get_metrics(y, y_pre):
        """
        compute the scores for a series of metrics
        :param y: reference array
        :param y_pre: candidate array
        :return: scores
        """
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        return hamming_loss, \
               macro_f1, macro_precision, macro_recall, \
               micro_f1, micro_precision, micro_recall

    # prepare the label sets to arrays
    y, y_pre = prepare_label(reference, candidate, label_dict)
    # score computation for a series of metrics
    # one_error = get_one_error(y, candidate, label_dict)
    hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall = get_metrics(
        y, y_pre)
    return {'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall}
