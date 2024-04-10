

from __future__ import print_function
import numpy as np
import torch
import random
from torch.autograd import Variable
from typing import Union, Any, List


from modules.reinforce.rouge import from_summary_index_compute_rouge, RougeScore
from modules.data.dataloader import Document


def is_termination(prob_masked: torch.Tensor):
    prob_masked = prob_masked.cpu().detach().numpy()
    maxnum = max(prob_masked)
    minnum = min(prob_masked)
    p = (maxnum - minnum)/maxnum
    p = max(p, maxnum)
    binary = np.random.binomial(1, p, size=(1))
    if binary:
        return False
    return True


class SummarizationReward:
    _SAMPLE_METHODS = ["lead3", "lead3_oracle", "greedy", "sample"]
    _RL_BASELINE_METHODS = ["batch_avg", "batch_med", "global_avg", "greedy"]
    _ROUGE_METRICS = ["avg_f", "avg_r", "f", "r", "all"]
    _EPS = 1e-8
    _EPSILON_BANDIT_POLICY = 0.1

    def __init__(self,
                 B: int = 20,
                 std_rouge: bool = False,
                 rouge_metric: str = "avg_f",
                 rl_baseline_method: str = "greedy") -> None:
        """_summary_

        Args:
            B (int, optional): _description_. Defaults to 20.
            std_rouge (bool, optional): _description_. Defaults to False.
            rouge_metric (str, optional): _description_. Value must be in ["avg_f", "avg_r", "f", "r", "all"]. Defaults to "avg_f".
            rl_baseline_method (str, optional): _description_. Value must be in ["batch_avg", "batch_med", "global_avg", "greedy"]. Defaults to "greedy".
        """
        assert rouge_metric in self._ROUGE_METRICS, f"rouge_metric must be in {self._ROUGE_METRICS}"
        assert rl_baseline_method in self._RL_BASELINE_METHODS, f"rl_baseline_method must be in {self._RL_BASELINE_METHODS}"

        self.B = B
        self.std_rouge = std_rouge
        self.rouge_metric = rouge_metric
        self.rl_baseline_method = rl_baseline_method

        self.probs_torch = None
        self.probs_numpy = None
        self.max_num_of_sents = None
        self.min_num_of_sents = None
        self.doc = None

        self.global_avg_reward = 0.
        self.train_examples_seen = 0.

    def update_data_instance(self,
                             probs: torch.Tensor,
                             doc: Document,
                             min_num_of_sents: int = 1,
                             max_num_of_sents: int = 3,):

        # Make sure probs don't contain zero
        self.probs_torch = probs * 0.999 + 0.001
        probs_numpy = probs.data.cpu().numpy()
        self.probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
        self.doc = doc

        self.min_num_of_sents = min(
            min_num_of_sents, self.probs_torch.size()[0])
        self.max_num_of_sents = min(
            self.probs_torch.size()[0], max_num_of_sents)

    @classmethod
    def get_summary_index(cls,
                          probs_numpy: np.ndarray,
                          probs_torch: torch.Tensor,
                          sample_method: str = "greedy",
                          if_test: bool = False,
                          max_num_of_sents: int = 4,
                          min_num_of_sents: int = 1):
        assert isinstance(sample_method, str)
        summary_index = []
        loss = None

        if max_num_of_sents <= 0:
            if sample_method == "sample":
                l = np.random.binomial(1, probs_numpy)
            elif sample_method == "greedy":
                l = [1 if prob >= 0.5 else 0 for prob in probs_numpy]
            summary_index = np.nonzero(l)[0]
        else:
            if sample_method == "sample":
                probs_torch = probs_torch.squeeze()
                assert len(probs_torch.size()) == 1
                if not if_test:
                    # herke's method
                    # epsilon = 0.1
                    mask = Variable(torch.ones(probs_torch.size()
                                               ).cuda(), requires_grad=False)
                    loss_list = []
                    for i in range(max_num_of_sents):
                        p_masked = probs_torch * mask
                        if random.uniform(0, 1) <= cls._EPSILON_BANDIT_POLICY:  # explore
                            # when testing, it should be closed
                            selected_idx = torch.multinomial(mask, 1)
                        else:
                            selected_idx = torch.multinomial(p_masked, 1)

                        # Compute loss func
                        loss_i = (cls._EPSILON_BANDIT_POLICY / mask.sum() + (1 - cls._EPSILON_BANDIT_POLICY) *
                                  p_masked[selected_idx] / p_masked.sum()).log()
                        loss_list.append(loss_i)
                        mask = mask.clone()
                        mask[selected_idx] = 0
                        summary_index.append(selected_idx)
                        if is_termination(p_masked) and len(summary_index) >= min_num_of_sents:
                            break
                    summary_index = torch.cat(summary_index, dim=0)
                    summary_index = summary_index.data.cpu().numpy()

                    loss = sum(loss_list)/(float(len(loss_list)) + cls._EPS)
                else:
                    mask = Variable(torch.ones(probs_torch.size()
                                               ).cuda(), requires_grad=False)
                    loss_list = []
                    for i in range(max_num_of_sents):
                        p_masked = probs_torch * mask

                        selected_idx = torch.multinomial(p_masked, 1)
                        mask = mask.clone()
                        mask[selected_idx] = 0
                        summary_index.append(selected_idx)
                        if is_termination(p_masked) and len(summary_index) >= min_num_of_sents:
                            break
                    summary_index = torch.cat(summary_index, dim=0)
                    summary_index = summary_index.data.cpu().numpy()

            elif sample_method == "greedy":
                loss = 0
                summary_index = np.argsort(np.reshape(
                    probs_numpy, len(probs_numpy)))[-max_num_of_sents:]
                summary_index = summary_index[::-1]

        # summary_index.sort()
        return summary_index, loss

    def sample_batch(self,
                     sample_method: str):
        batch_index_and_loss_lists = [
            self.generate_indexes_and_loss(self.probs_numpy,
                                           self.probs_torch,
                                           sample_method=sample_method,
                                           max_num_of_sents=self.max_num_of_sents,
                                           min_num_of_sents=self.min_num_of_sents) for i in range(self.B)]
        return batch_index_and_loss_lists

    @classmethod
    def generate_indexes_and_loss(cls,
                                  probs_numpy: np.ndarray,
                                  probs_torch: torch.Tensor,
                                  sample_method: str = "sample",
                                  min_num_of_sents: int = 1,
                                  max_num_of_sents: int = 3):

        if sample_method == "lead3":
            return range(3), 0
        elif sample_method == "lead_oracle":
            return range(max_num_of_sents), 0
        else:  # either "sample" or "greedy" based on the prob_list
            return cls.get_summary_index(probs_numpy,
                                         probs_torch,
                                         sample_method=sample_method,
                                         max_num_of_sents=max_num_of_sents,
                                         min_num_of_sents=min_num_of_sents)

    def get_reward(self,
                   summary_index: Union[list, Any],
                   max_num_of_bytes: int = -1):
        reward = from_summary_index_compute_rouge(self.doc,
                                                  summary_index,
                                                  std_rouge=self.std_rouge,
                                                  rouge_metric=self.rouge_metric,
                                                  max_num_of_bytes=max_num_of_bytes)
        return reward

    def compute_baseline(self,
                         batch_rewards: list,
                         max_num_of_bytes: int = -1):
        def running_avg(t, old_mean, new_score):
            return (t - 1) / t * old_mean + new_score / t

        batch_avg_reward = np.mean(batch_rewards)
        batch_median_reward = np.median(batch_rewards)
        self.global_avg_reward = running_avg(
            self.train_examples_seen, self.global_avg_reward, batch_avg_reward)

        if self.rl_baseline_method == "batch_avg":
            return batch_avg_reward
        if self.rl_baseline_method == "batch_med":
            return batch_median_reward
        elif self.rl_baseline_method == "global_avg":
            return self.global_avg_reward
        elif self.rl_baseline_method == "greedy":
            summary_index_list, _ = self.generate_indexes_and_loss(sample_method="greedy",
                                                                   probs_numpy=self.probs_numpy,
                                                                   probs_torch=self.probs_torch,
                                                                   min_num_of_sents=self.min_num_of_sents,
                                                                   max_num_of_sents=self.max_num_of_sents)
            reward = self.get_reward(
                summary_index_list, max_num_of_bytes)
            return reward
        else:  # none
            return 0

    def generate_batch_loss(self,
                            batch_index_and_loss_lists: List[List[Any]],
                            batch_rewards: List[Any],
                            rl_baseline_reward: Any):

        losses = []
        for i in range(len(batch_rewards)):
            loss = batch_index_and_loss_lists[i][1] * (
                (rl_baseline_reward - batch_rewards[i]) / (rl_baseline_reward + self._EPS))
            losses.append(loss)

        avg_loss = sum(loss) / (float(len(loss)) + self._EPS)
        return avg_loss

    def train(self,
              probs: torch.Tensor,
              doc: Document,
              min_num_of_sents: int = 1,
              max_num_of_sents: int = 3,
              max_num_of_bytes: int = -1):
        """
        :return: training_loss_of_the current example
        """
        self.update_data_instance(
            probs, doc, min_num_of_sents, max_num_of_sents)

        self.train_examples_seen += 1
        # sample 20 times
        batch_index_and_loss_lists = self.sample_batch(sample_method="sample")

        # 20 rewards
        batch_rewards = [
            self.get_reward(idx_list[0], max_num_of_bytes)
            for idx_list in batch_index_and_loss_lists
        ]

        # mean rewards of 20 samples
        rl_baseline_reward = self.compute_baseline(batch_rewards,
                                                   )
        # compute loss with rewards and cross entropy
        loss = self.generate_batch_loss(
            batch_index_and_loss_lists, batch_rewards, rl_baseline_reward)

        greedy_index_list, _ = self.generate_indexes_and_loss(sample_method="greedy",
                                                              probs_numpy=self.probs_numpy,
                                                              probs_torch=self.probs_torch,
                                                              min_num_of_sents=self.min_num_of_sents,
                                                              max_num_of_sents=self.max_num_of_sents)
        greedy_reward = self.get_reward(greedy_index_list,
                                        max_num_of_bytes)

        return loss, greedy_reward


class ReinforceLoss:
    _sample_method = "greedy"

    @classmethod
    def get(cls,
            probs: torch.Tensor,
            doc: Document,
            rouge_metric: str = "all",
            max_num_of_sents: int = 3,
            max_num_of_bytes: int = -1,
            min_num_of_sents: int = 1,
            std_rouge: bool = False,
            test: bool = False,
            score_flag: bool = False,
            path: str = "./"
            ):
        # sample sentences
        probs_numpy = probs.data.cpu().numpy()
        probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
        # max of sents# in doc and sents# in summary
        max_num_of_sents = min(len(probs_numpy), max_num_of_sents)

        rl_baseline_summary_index, _ = SummarizationReward.get_summary_index(probs_numpy,
                                                                             probs,
                                                                             sample_method=cls._sample_method,
                                                                             min_num_of_sents=min_num_of_sents,
                                                                             max_num_of_sents=max_num_of_sents,
                                                                             if_test=test)

        rl_baseline_summary_index = sorted(rl_baseline_summary_index)

        rl_baseline_reward = RougeScore.from_summary_index_and_compute_rouge(
            doc, rl_baseline_summary_index, rouge_metric=rouge_metric,
            std_rouge=std_rouge,
            max_num_of_bytes=max_num_of_bytes,
            score_flag=score_flag,
            path=path)

        lead3_reward = RougeScore.from_summary_index_and_compute_rouge(
            doc, range(max_num_of_sents),
            rouge_metric=rouge_metric,
            std_rouge=std_rouge,
            max_num_of_bytes=max_num_of_bytes,
            score_flag=score_flag,
            path=path)


        return rl_baseline_reward, lead3_reward

if __name__ == '__main__':
    pass
