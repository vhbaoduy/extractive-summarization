import argparse

from modules.data.dataloader import CNNDailyMailDataset, BatchDataLoader
from modules.data.vocab import Vocab
from modules.models.HER import HERExtractor
from modules.reinforce.reward import ReinforceReward, SummarizationReward
from modules.reinforce.rouge import reinforce_loss


from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import logging
from typing import Any
import pandas as pd
import datetime as dt


def eval_model(extractor: torch.nn.Module,
               dataloader: BatchDataLoader,
               eval_data: str = "test",
               std_rouge: bool = False,
               length_limit: int = -1,
               rouge_metric: str = "all",
               oracle_length: int = -1):
    extractor.eval()
    # eval_reward, lead3_reward = evaluate.ext_model_eval(
    #     extractor, vocab, args, "val")

    progress_bar = tqdm.tqdm(enumerate(dataloader),
                             dynamic_ncols=True, total=len(dataloader))
    messsage = '[Eval] - Mean reward %.4f Lead3  %.4f'
    eval_rewards, lead3_rewards = [], []
    step_in_epoch = 0
    for step, batch_data in progress_bar:
        for doc in batch_data:
            try:
                # if True:
                step_in_epoch += 1
                # for i in range(1):  # how many times a single data gets updated before proceeding
                if oracle_length == -1:  # use true oracle length
                    oracle_summary_sent_num = len(doc.tokenized_summary)
                else:
                    oracle_summary_sent_num = oracle_length

                x = doc.tokenized_content
                if min(x.shape) == 0:
                    continue
                sents = Variable(torch.from_numpy(x)).cuda()
                outputs = extractor(sents)
                compute_score = (step == len(dataloader.dataset) -
                                 1) or (std_rouge is False)
                if eval_data == "test":
                    reward, lead3_r = reinforce_loss(outputs, doc, id=doc.id,
                                                     max_num_of_sents=oracle_summary_sent_num,
                                                     max_num_of_bytes=length_limit,
                                                     std_rouge=True,
                                                     rouge_metric="all",
                                                     compute_score=compute_score)
                else:
                    reward, lead3_r = reinforce_loss(outputs, doc, id=doc.id,
                                                     max_num_of_sents=oracle_summary_sent_num,
                                                     max_num_of_bytes=length_limit,
                                                     std_rouge=std_rouge,
                                                     rouge_metric=rouge_metric,
                                                     compute_score=compute_score)

                eval_rewards.append(reward)
                lead3_rewards.append(lead3_r)
                progress_bar.set_description(messsage % (
                    np.mean(eval_rewards), np.mean(lead3_rewards)))
            except Exception as e:
                print(f"Erro when training {e}")
    avg_eval_r = np.mean(eval_rewards, axis=0)
    avg_lead3_r = np.mean(lead3_rewards, axis=0)
    # print('epoch ' + str(epoch) + ' reward in validation: '
    #         + str(eval_reward) + ' lead3: ' + str(lead3_reward))
    return avg_eval_r, avg_lead3_r


if __name__ == "__main__":
    vocab = Vocab(vocab_file_path="data/vocab",
                  glove_file_path="data/vocab_100d.txt",
                  embedding_size=100)

    df_train = pd.read_csv("data/cnn_dailymail/train.csv")
    df_val = pd.read_csv("data/cnn_dailymail/valid.csv")

    df_train = df_train
    df_val = df_val
    train_dataset = CNNDailyMailDataset(df=df_train,
                                        word2id=vocab)
    train_dataloader = BatchDataLoader(train_dataset,
                                       batch_size=1,
                                       shuffle=True)

    val_dataset = CNNDailyMailDataset(df=df_val,
                                      word2id=vocab)
    val_dataloader = BatchDataLoader(val_dataset,
                                     batch_size=1,
                                     shuffle=True)

    vocab_embeddings = vocab.get_embedding()
    vocab_size, emb_dims = vocab_embeddings.shape
    extractor = HERExtractor(
        vocab_size=vocab_size,
        embedding_dim=emb_dims,
        word_input_size=100,
        sentence_input_size=400,
        decode_hidden_units=200,
        dropout_p=0,
        num_lstm_layers=2,
        num_filters=100,
        word_lstm_hidden_units=200,
        sentence_lstm_hidden_units=200,
        word_pretrained_embedding=vocab_embeddings
    )
    # outputs = extractor(sents)
    extractor = extractor.cuda()

    model_name = "HER_extractor.best.ext"

    log_name = "logs/model_log_" + \
        dt.datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + ".log"

    # Args
    lr = 1e-5
    epochs_ext = 1
    oracle_length = -1
    length_limit = -1

    # print("Hello")
    optimizer_ext = torch.optim.Adam(
        extractor.parameters(), lr=lr, betas=(0., 0.999))
    print("starting training")
    n_step = 200

    # init statistics
    reward_list = []
    best_eval_reward = 0.

    logging.basicConfig(handlers=[logging.FileHandler('%s.log' % log_name), logging.StreamHandler()],
                        level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    reinforce = SummarizationReward(B=20,
                                    rouge_metric="avg_f",
                                    rl_baseline_method="batch_avg")
    messsage = '[Train] - Epoch %d Step %d Reward %.4f'
    for epoch in range(epochs_ext):
        step_in_epoch = 0
        progress_bar = tqdm.tqdm(
            enumerate(train_dataloader), dynamic_ncols=True, total=len(train_dataloader))
        for step, batch_data in progress_bar:
            for doc in batch_data:
                try:
                    extractor.train()
                    # if True:
                    step_in_epoch += 1
                    # for i in range(1):  # how many times a single data gets updated before proceeding
                    if oracle_length == -1:  # use true oracle length
                        oracle_summary_sent_num = len(doc.tokenized_summary)
                    else:
                        oracle_summary_sent_num = oracle_length

                    x = doc.tokenized_content
                    if min(x.shape) == 0:
                        continue
                    sents = Variable(torch.from_numpy(x)).cuda()
                    outputs = extractor(sents)

                    loss, reward = reinforce.train(outputs,
                                                   doc,
                                                   max_num_of_sents=oracle_summary_sent_num,
                                                   max_num_of_bytes=length_limit,)

                    reward_list.append(reward)

                    if isinstance(loss, Variable):
                        loss.backward()

                    if step % 1 == 0:
                        torch.nn.utils.clip_grad_norm_(
                            extractor.parameters(), 1)  # gradient clipping
                        optimizer_ext.step()
                        optimizer_ext.zero_grad()
                    # print('Epoch %d Step %d Reward %.4f' %
                    #     (epoch, step_in_epoch, reward))
                    progress_bar.set_description(
                        messsage % (epoch, step_in_epoch, reward))
                except Exception as e:
                    print(f"Erro when training {e}")

                if (step_in_epoch) % n_step == 0 and step_in_epoch != 0:
                    logging.info('[Train] - Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                                 ' Mean reward: ' + str(np.mean(reward_list)))
                    reward_list = []

            if len(train_dataset) == step_in_epoch or ((step_in_epoch) % 1000 == 0 and step_in_epoch != 0):
                eval_reward, lead3_reward = eval_model(
                    extractor, val_dataloader,
                    eval_data="val",
                    std_rouge=False,
                    length_limit=length_limit,
                    rouge_metric="avg_f",)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    logging.info("Saving model %s with eval_reward: %s leadreward: %s" %
                                 (model_name, eval_reward, lead3_reward))
                    torch.save(extractor, model_name)
                logging.info('[Eval] - Epoch ' + str(epoch) +
                             ' Mean reward: ' + str(np.mean(eval_reward)))
