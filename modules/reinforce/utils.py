from modules.data.dataloader import Document
import numpy as np
import torch
from torch.autograd import Variable
from modules.data.dataloader import Document
import random

method = 'herke'


def cutwords(sens, max_num_of_chars):
    output = []
    quota = max_num_of_chars
    for sen in sens:
        if quota > len(sen):
            output.append(sen)
            quota -= len(sen)
        else:
            output.append(sen[:quota])
            break
    return output


def from_summary_index_generate_hyp_ref(doc: Document, summary_index: list):
    hyp = doc.get_sentences(summary_index)
    ref = doc.get_summary()
    return hyp, ref


def get_summary_index(probs_numpy,
                      probs_torch,
                      sample_method: str = "greedy",
                      max_num_of_sents: int = 3):
    assert isinstance(sample_method, str)
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

            if method == 'original':
                # original method
                probs_clip = probs_numpy * 0.8 + 0.1
                # print("sampling the index for the summary")
                index = range(len(probs_clip))
                probs_clip_norm = probs_clip / sum(probs_clip)
                summary_index = np.random.choice(index, max_num_of_sents, replace=False,
                                                 p=np.reshape(probs_clip_norm, len(probs_clip_norm)))
                p_summary_index = probs_numpy[summary_index]
                sorted_idx = np.argsort(p_summary_index)[::-1]
                summary_index = summary_index[sorted_idx]
                loss = 0.
                for idx in index:
                    if idx in summary_index:
                        loss += probs_torch[idx].log()
                    else:
                        loss += (1 - probs_torch[idx]).log()
            elif method == 'herke':
                # herke's method
                summary_index = []
                epsilon = 0.1
                mask = Variable(torch.ones(probs_torch.size()
                                           ).cuda(), requires_grad=False)
                loss_list = []
                for i in range(max_num_of_sents):
                    p_masked = probs_torch * mask
                    if random.uniform(0, 1) <= epsilon:  # explore
                        selected_idx = torch.multinomial(mask, 1)
                    else:
                        selected_idx = torch.multinomial(p_masked, 1)
                    loss_i = (epsilon / mask.sum() + (1 - epsilon) *
                              p_masked[selected_idx] / p_masked.sum()).log()
                    loss_list.append(loss_i)
                    mask = mask.clone()
                    mask[selected_idx] = 0
                    summary_index.append(selected_idx)

                summary_index = torch.cat(summary_index, dim=0)
                summary_index = summary_index.data.cpu().numpy()

                loss = sum(loss_list)
        elif sample_method == "greedy":
            loss = 0
            summary_index = np.argsort(np.reshape(
                probs_numpy, len(probs_numpy)))[-max_num_of_sents:]
            summary_index = summary_index[::-1]

    # summary_index.sort()
    return summary_index, loss
