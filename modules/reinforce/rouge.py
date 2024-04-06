import codecs
import os
import shutil
import torch
import json

from pyrouge import Rouge155
from rouge import Rouge
from modules.data.dataloader import Document
import numpy as np
from modules.reinforce.utils import cutwords, get_summary_index, from_summary_index_generate_hyp_ref

rouge = Rouge()
rouge155 = Rouge155('ROUGE-1.5.5/',
                    '-e ROUGE-1.5.5/data -a -c 95 -m -n 2 -b %d' % (-1))

PATH_RESULT = "./result"


class RougeScore:
    @classmethod
    def pyrouge_score(cls,
                      path_eval: str,
                      save_path: str):
        rouge155.system_dir = path_eval
        rouge155.model_dir = path_eval
        rouge155.system_filename_pattern = 'hyp.(\d+).txt'
        rouge155.model_filename_pattern = 'ref.#ID#.txt'

        output = rouge155.evaluate()
        output_dict = rouge155.output_to_dict(output)
        # cleanup
        # shutil.rmtree(path_eval)
        # shutil.rmtree(rouge155._config_dir)
        json.dump(output_dict, open(save_path, "w"))

        return (output_dict["rouge_1_precision"], output_dict["rouge_1_recall"], output_dict["rouge_1_f_score"],
                output_dict["rouge_2_precision"], output_dict["rouge_2_recall"], output_dict["rouge_2_f_score"],
                output_dict["rouge_l_precision"], output_dict["rouge_l_recall"], output_dict["rouge_l_f_score"])

    @classmethod
    def compute_score(cls, ref, hyp,
                      rouge_metric, max_num_of_bytes=-1,
                      id: str = "",
                      std_rouge=False,
                      score_flag=True,
                      path=None):
        if not std_rouge:
            ref = [_.lower() for _ in ref]
            # join for managing the cases where we have different number of sentence.
            ref = [" ".join(ref)]
            hyp = [_.lower().replace(".", " .") for _ in hyp]
            hyp = [" ".join(hyp)]

            if max_num_of_bytes > 0:
                ref = cutwords(ref)
                hyp = cutwords(hyp)

            rouge_score = rouge.get_scores(hyp, ref)
            if rouge_metric[1] == 'f':
                return rouge_score[0]['rouge-%s' % rouge_metric[0]]['f']
            elif rouge_metric[1] == 'r':
                return rouge_score[0]['rouge-%s' % rouge_metric[0]]['r']
            elif rouge_metric == 'avg_f':
                return (rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 3
            elif rouge_metric == 'avg_r':
                return (rouge_score[0]['rouge-1']['r'] + rouge_score[0]['rouge-2']['r'] + rouge_score[0]['rouge-l']['r']) / 3
            else:
                return (rouge_score[0]['rouge-1']['p'], rouge_score[0]['rouge-1']['r'], rouge_score[0]['rouge-1']['f'],
                        rouge_score[0]['rouge-2']['p'], rouge_score[0]['rouge-2']['r'], rouge_score[0]['rouge-2']['f'],
                        rouge_score[0]['rouge-l']['p'], rouge_score[0]['rouge-l']['r'], rouge_score[0]['rouge-l']['f'])

        else:
            # initialization
            if not os.path.exists(PATH_RESULT):
                os.mkdir(PATH_RESULT)
            if not os.path.exists(path):
                os.mkdir(path)

            path_eval = os.path.join(path, "eval")
            os.makedirs(path_eval, exist_ok=True)
            # write new ref and hyp
            with codecs.open(os.path.join(path_eval, 'ref.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
                f.write(Rouge155.convert_text_to_rouge_format('\n'.join(ref)))
            with codecs.open(os.path.join(path_eval, 'hyp.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
                f.write(Rouge155.convert_text_to_rouge_format('\n'.join(hyp)))
            # if score_flag:
            #     rouge155.system_dir = path_eval
            #     rouge155.model_dir = path_eval
            #     rouge155.system_filename_pattern = 'hyp.(\d+).txt'
            #     rouge155.model_filename_pattern = 'ref.#ID#.txt'

            #     output = rouge155.evaluate()
            #     output_dict = rouge155.output_to_dict(output)
            #     # cleanup
            #     shutil.rmtree(path_eval)
            #     shutil.rmtree(rouge155._config_dir)

            #     if rouge_metric[1] == 'f':
            #         return output_dict["rouge_%s_f_score" % rouge_metric[0]]
            #     elif rouge_metric[1] == 'r':
            #         return output_dict["rouge_%s_recall" % rouge_metric[0]]
            #     elif rouge_metric == 'avg_f':
            #         return (output_dict["rouge_1_f_score"] + output_dict["rouge_2_f_score"] + output_dict[
            #             "rouge_l_f_score"]) / 3
            #     elif rouge_metric == 'avg_r':
            #         return (output_dict["rouge_1_recall"] + output_dict["rouge_2_recall"] + output_dict["rouge_l_recall"]) / 3
            #     else:
            #         return (output_dict["rouge_1_precision"], output_dict["rouge_1_recall"], output_dict["rouge_1_f_score"],
            #                 output_dict["rouge_2_precision"], output_dict["rouge_2_recall"], output_dict["rouge_2_f_score"],
            #                 output_dict["rouge_l_precision"], output_dict["rouge_l_recall"], output_dict["rouge_l_f_score"])
            # else:
            return 0.0

    @classmethod
    def from_summary_index_and_compute_rouge(cls,
                                             doc: Document,
                                             summary_index: list,
                                             std_rouge=False,
                                             rouge_metric="all",
                                             max_num_of_bytes=-1,
                                             score_flag=False,
                                             path="./"):
        # greedy approach directly use this
        hyp = doc.get_sentences(summary_index)
        ref = doc.get_summary()
        if len(hyp) == 0 or len(ref) == 0:
            return 0.
        # if std_rouge:
        #     score = RougeTest_pyrouge(ref, hyp, rouge_metric=rouge_metric)
        # else:
        #     score = RougeTest_rouge(
        #         ref, hyp, rouge_metric=rouge_metric, max_num_of_bytes=max_num_of_bytes)
        score = cls.compute_score(ref, hyp,
                                  id=doc.index,
                                  rouge_metric=rouge_metric,
                                  max_num_of_bytes=max_num_of_bytes,
                                  std_rouge=std_rouge,
                                  score_flag=score_flag,
                                  path=path)
        return score


def RougeTest_rouge(ref, hyp, rouge_metric="all", max_num_of_bytes=-1):
    ref = [_.lower() for _ in ref]
    # join for managing the cases where we have different number of sentence.
    ref = [" ".join(ref)]
    hyp = [_.lower().replace(".", " .") for _ in hyp]
    hyp = [" ".join(hyp)]

    if max_num_of_bytes > 0:
        ref = cutwords(ref)
        hyp = cutwords(hyp)

    rouge_score = rouge.get_scores(hyp, ref)
    if rouge_metric[1] == 'f':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['f']
    elif rouge_metric[1] == 'r':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['r']
    elif rouge_metric == 'avg_f':
        return (rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 3
    elif rouge_metric == 'avg_r':
        return (rouge_score[0]['rouge-1']['r'] + rouge_score[0]['rouge-2']['r'] + rouge_score[0]['rouge-l']['r']) / 3
    else:
        return (rouge_score[0]['rouge-1']['p'], rouge_score[0]['rouge-1']['r'], rouge_score[0]['rouge-1']['f'],
                rouge_score[0]['rouge-2']['p'], rouge_score[0]['rouge-2']['r'], rouge_score[0]['rouge-2']['f'],
                rouge_score[0]['rouge-l']['p'], rouge_score[0]['rouge-l']['r'], rouge_score[0]['rouge-l']['f'])


# home_path = "./data"
rouge155 = Rouge155('ROUGE-1.5.5/',
                    '-e ROUGE-1.5.5/data -a -c 95 -m -n 2 -b %d' % (-1))


def RougeTest_pyrouge(ref, hyp, id=0, rouge_metric='all', compute_score=True,
                      path='./result'):
    # initialization
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(path):
        os.mkdir(path)
    # write new ref and hyp
    with codecs.open(os.path.join(path, 'ref.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
        f.write(Rouge155.convert_text_to_rouge_format('\n'.join(ref)))
    with codecs.open(os.path.join(path, 'hyp.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
        f.write(Rouge155.convert_text_to_rouge_format('\n'.join(hyp)))

    # if compute_score:
    #     rouge155.system_dir = path
    #     rouge155.model_dir = path
    #     rouge155.system_filename_pattern = 'hyp.(\d+).txt'
    #     rouge155.model_filename_pattern = 'ref.#ID#.txt'

    #     output = rouge155.evaluate()
    #     output_dict = rouge155.output_to_dict(output)
    #     # cleanup
    #     shutil.rmtree(path)
    #     shutil.rmtree(rouge155._config_dir)

    #     if rouge_metric[1] == 'f':
    #         return output_dict["rouge_%s_f_score" % rouge_metric[0]]
    #     elif rouge_metric[1] == 'r':
    #         return output_dict["rouge_%s_recall" % rouge_metric[0]]
    #     elif rouge_metric == 'avg_f':
    #         return (output_dict["rouge_1_f_score"] + output_dict["rouge_2_f_score"] + output_dict[
    #             "rouge_l_f_score"]) / 3
    #     elif rouge_metric == 'avg_r':
    #         return (output_dict["rouge_1_recall"] + output_dict["rouge_2_recall"] + output_dict["rouge_l_recall"]) / 3
    #     else:
    #         return (output_dict["import src.dataLoader"], output_dict["rouge_1_recall"], output_dict["rouge_1_f_score"],
    #                 output_dict["rouge_2_precision"], output_dict["rouge_2_recall"], output_dict["rouge_2_f_score"],
    #                 output_dict["rouge_l_precision"], output_dict["rouge_l_recall"], output_dict["rouge_l_f_score"])
    # else:
    #     return None


def from_summary_index_compute_rouge(doc, summary_index, std_rouge=False, rouge_metric="all", max_num_of_bytes=-1):
    # greedy approach directly use this

    hyp, ref = from_summary_index_generate_hyp_ref(doc, summary_index)
    if len(hyp) == 0 or len(ref) == 0:
        return 0.

    if std_rouge:
        score = RougeTest_pyrouge(ref, hyp, rouge_metric=rouge_metric)
    else:
        score = RougeTest_rouge(
            ref, hyp, rouge_metric=rouge_metric, max_num_of_bytes=max_num_of_bytes)
    return score


def reinforce_loss(probs, doc, id=0,
                   max_num_of_sents=3, max_num_of_bytes=-1,
                   std_rouge=False, rouge_metric="all", compute_score=True):
    # sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    # max of sents# in doc and sents# in summary
    max_num_of_sents = min(len(probs_numpy), max_num_of_sents)

    rl_baseline_summary_index, _ = get_summary_index(probs_numpy, probs,
                                                     sample_method="greedy", max_num_of_sents=max_num_of_sents)
    rl_baseline_summary_index = sorted(rl_baseline_summary_index)
    rl_baseline_hyp, rl_baseline_ref = from_summary_index_generate_hyp_ref(
        doc, rl_baseline_summary_index)

    lead3_hyp, lead3_ref = from_summary_index_generate_hyp_ref(
        doc, range(max_num_of_sents))
    if std_rouge:
        rl_baseline_reward = RougeTest_pyrouge(rl_baseline_ref, rl_baseline_hyp, id=id, rouge_metric=rouge_metric,
                                               compute_score=compute_score, path=os.path.join('./result/rl'))
        lead3_reward = RougeTest_pyrouge(lead3_ref, lead3_hyp, id=id, rouge_metric=rouge_metric,
                                         compute_score=compute_score, path=os.path.join('./result/lead'))
    else:
        rl_baseline_reward = RougeTest_rouge(rl_baseline_ref, rl_baseline_hyp, rouge_metric,
                                             max_num_of_bytes=max_num_of_bytes)
        lead3_reward = RougeTest_rouge(
            lead3_ref, lead3_hyp, rouge_metric, max_num_of_bytes=max_num_of_bytes)

    return rl_baseline_reward, lead3_reward
