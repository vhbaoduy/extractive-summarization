import argparse

from modules.data.dataloader import CNNDailyMailDataset, BatchDataLoader
from modules.data.vocab import Vocab
from modules.models.HER import HERExtractor

from modules.reinforce.reward import SummarizationReward, ReinforceLoss
from modules.reinforce.rouge import RougeScore


from torch.autograd import Variable
import torch
import tqdm
import numpy as np
import logging
from typing import Any
import pandas as pd
import datetime as dt
import os
import yaml
import json

from utils.config import Configs, ModelConfig
import sys
sys.path.append(".")

logging.getLogger('global').disabled = True


def eval_extractor(extractor: torch.nn.Module,
                   dataloader: BatchDataLoader,
                   configs: Configs,
                   eval_data: str = "test",
                   device="cuda"):
    extractor.eval()
    # eval_reward, lead3_reward = evaluate.ext_model_eval(
    #     extractor, vocab, args, "val")

    progress_bar = tqdm.tqdm(enumerate(dataloader),
                             dynamic_ncols=True, total=len(dataloader))
    messsage = '[Eval] - Mean reward %.4f Lead3  %.4f'
    eval_rewards, lead3_rewards = [], []
    step_in_epoch = 0
    rouge_metric = configs.reinforce.rouge_metric
    std_rouge = configs.reinforce.std_rouge
    compute_score = False
    if eval_data == "test":
        rouge_metric = "all"
        std_rouge = False
        compute_score = False
    for step, batch_data in progress_bar:
        for doc in batch_data:
            try:
                # if True:
                step_in_epoch += 1
                if len(doc.sentences) == 0 or len(doc.tokenized_summary) == 0:
                    continue
                if len(doc.sentences) < max(configs.model.kernel_sizes):
                    summary_index_list = range(min(len(doc.sentences), 3))

                    reward = RougeScore.from_summary_index_and_compute_rouge(doc, summary_index_list,
                                                                             std_rouge=std_rouge,
                                                                             rouge_metric=rouge_metric,
                                                                             max_num_of_bytes=configs.reinforce.length_limit,
                                                                             path=configs.result_path,
                                                                             score_flag=compute_score)
                    lead3_r = reward
                else:
                    # for i in range(1):  # how many times a single data gets updated before proceeding
                    if configs.reinforce.oracle_length == -1:  # use true oracle length
                        oracle_summary_sent_num = len(doc.tokenized_summary)
                    else:
                        oracle_summary_sent_num = configs.reinforce.oracle_length

                    x = doc.tokenized_content
                    if min(x.shape) == 0:
                        continue
                    sents = Variable(torch.from_numpy(x)).to(device)
                    outputs = extractor(sents)

                    if eval_data == "test":
                        reward, lead3_r = ReinforceLoss.get(probs=outputs,
                                                            doc=doc,
                                                            rouge_metric=rouge_metric,
                                                            max_num_of_sents=oracle_summary_sent_num,
                                                            max_num_of_bytes=configs.reinforce.length_limit,
                                                            std_rouge=std_rouge,
                                                            path=configs.result_path,
                                                            score_flag=True,
                                                            test=True)
                    else:
                        reward, lead3_r = ReinforceLoss.get(probs=outputs,
                                                            doc=doc,
                                                            rouge_metric=rouge_metric,
                                                            max_num_of_sents=oracle_summary_sent_num,
                                                            max_num_of_bytes=configs.reinforce.length_limit,
                                                            std_rouge=std_rouge,
                                                            score_flag=compute_score,
                                                            path=configs.result_path)

                eval_rewards.append(reward)
                lead3_rewards.append(lead3_r)
                progress_bar.set_description(messsage % (
                    np.mean(eval_rewards), np.mean(lead3_rewards)))
            except Exception as e:
                logging.error(f"Error when eval {e}")

    avg_eval_r = np.mean(eval_rewards, axis=0)
    avg_lead3_r = np.mean(lead3_rewards, axis=0)
    # print('epoch ' + str(epoch) + ' reward in validation: '
    #         + str(eval_reward) + ' lead3: ' + str(lead3_reward))
    return avg_eval_r, avg_lead3_r


def train_extractor(extractor: torch.nn.Module,
                    train_dataloader: BatchDataLoader,
                    val_dataloader: BatchDataLoader,
                    configs: Configs,
                    device: str = "cuda"):

    extractor = extractor.to(device)

    optimizer_ext = torch.optim.Adam(extractor.parameters(),
                                     lr=configs.optimize.lr,
                                     betas=configs.optimize.beta,
                                     weight_decay=configs.optimize.weight_decay)
    logging.info("Starting training")

    model_name = "best_model_exp_" + \
        dt.datetime.now().strftime("%Y-%m-%d.%H_%M_%S") + ".pt"
    path_to_save_model = os.path.join(configs.save_path, model_name)
    # init statistics
    reward_list = []
    best_eval_reward = 0.

    reinforce = SummarizationReward(B=configs.reinforce.B,
                                    rouge_metric=configs.reinforce.rouge_metric,
                                    std_rouge=configs.reinforce.std_rouge,
                                    rl_baseline_method=configs.reinforce.rl_baseline_method)
    messsage = '[Train] - Epoch %d Step %d Reward %.4f'
    for epoch in range(configs.optimize.epochs):
        step_in_epoch = 0
        progress_bar = tqdm.tqdm(
            enumerate(train_dataloader), dynamic_ncols=True, total=len(train_dataloader))
        for step, batch_data in progress_bar:
            for doc in batch_data:
                try:
                    extractor.train()
                    # if True:
                    step_in_epoch += 1
                    if len(doc.sentences) == 0 or len(doc.tokenized_summary) == 0:
                        continue

                    if len(doc.sentences) < max(configs.model.kernel_sizes):
                        summary_index_list = range(min(len(doc.sentences), 3))
                        loss = 0
                        reward = RougeScore.from_summary_index_and_compute_rouge(doc, summary_index_list,
                                                                                 std_rouge=configs.reinforce.std_rouge,
                                                                                 rouge_metric=configs.reinforce.rouge_metric,
                                                                                 max_num_of_bytes=configs.reinforce.length_limit)
                    else:
                        if configs.reinforce.oracle_length == -1:  # use true oracle length
                            oracle_summary_sent_num = len(
                                doc.tokenized_summary)
                        else:
                            oracle_summary_sent_num = configs.reinforce.oracle_length

                        x = doc.tokenized_content
                        if min(x.shape) == 0:
                            continue

                        sents = Variable(torch.from_numpy(x)).to(device)
                        outputs = extractor(sents)
                        loss, reward = reinforce.train(outputs,
                                                       doc,
                                                       max_num_of_sents=oracle_summary_sent_num,
                                                       max_num_of_bytes=configs.reinforce.length_limit)

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

                    reward_list.append(reward)
                except Exception as e:
                    logging.error(f"Erro when training {e}")

                if (step_in_epoch) % configs.optimize.print_steps == 0 and step_in_epoch != 0:
                    logging.info('[Train] - Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                                 ' Mean reward: ' + str(np.mean(reward_list)))
                    reward_list = []

            if len(train_dataloader.dataset) == step_in_epoch \
                    or ((step_in_epoch) % configs.optimize.eval_steps == 0 and step_in_epoch != 0):
                eval_reward, lead3_reward = eval_extractor(
                    extractor, val_dataloader,
                    eval_data="val",
                    configs=configs)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    logging.info("Saving model %s with eval_reward: %s leadreward: %s" %
                                 (path_to_save_model, eval_reward, lead3_reward))
                    torch.save(extractor, path_to_save_model)
                logging.info('[Eval] - Epoch ' + str(epoch) +
                             ' Mean reward: ' + str(np.mean(eval_reward)))

    return extractor


def get_extractor(config: ModelConfig,
                  vocab_size: int,
                  embedding_dim: int,
                  pretrained_embeddings: Any):
    extractor = None
    if config.extractor == "HER":
        extractor = HERExtractor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            word_input_size=config.word_input_size,
            sentence_input_size=config.sentence_input_size,
            decode_hidden_units=config.decode_hidden_units,
            dropout_p=config.dropout,
            num_lstm_layers=config.num_lstm_layers,
            word_lstm_hidden_units=config.word_lstm_hidden_units,
            sentence_lstm_hidden_units=config.sentence_lstm_hidden_units,
            word_pretrained_embedding=pretrained_embeddings
        )
    return extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str,
                        default='./data/vocab')
    parser.add_argument('--glove_file', type=str,
                        default='./data/vocab_100d.txt')
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--data_dir', type=str,
                        default='./data/cnn_dailymail/')

    parser.add_argument('--config_path', type=str,
                        default='./configs/exp.yaml')

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--eval_mode', type=bool, default=True)
    parser.add_argument('--eval_data', type=str, default="valid")
    parser.add_argument('--pretrained_model', type=str,
                        default="data/best_model_2.pt")

    args = parser.parse_args()

    configs = None
    config_json = {}
    with open(args.config_path, "r") as f:
        config_json = yaml.load(f, yaml.SafeLoader)
    configs = Configs(**config_json)

    # logger = logging.getLogger(name="exp")
    log_name = "exp_log_" + dt.datetime.now().strftime("%Y-%m-%d")
    path_to_log_file = os.path.join(configs.log_path, log_name)
    # logger.addHandler(logging.FileHandler('%s.log' % path_to_log_file))
    # logger.addHandler(logging.StreamHandler())
    # logger.info("Hello")

    os.makedirs(configs.log_path, exist_ok=True)
    os.makedirs(configs.save_path, exist_ok=True)
    os.makedirs(configs.result_path, exist_ok=True)

    logging.basicConfig(handlers=[logging.FileHandler('%s.log' % path_to_log_file), logging.StreamHandler()],
                        level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
    logging.info(json.dumps(config_json, indent=4))

    if not args.eval_mode:

        vocab = Vocab(vocab_file_path=args.vocab_file,
                      glove_file_path=args.glove_file,
                      embedding_size=args.embedding_size)
        vocab_embeddings = vocab.get_embedding()
        vocab_size, emb_dims = vocab_embeddings.shape

        extractor = None
        if args.pretrained_model:
            logging.info(f"Load model from {args.pretrained_model}")
            extractor = torch.load(args.pretrained_model,
                                   map_location=lambda storage, loc: storage)
        extractor = get_extractor(configs.model,
                                  vocab_size=vocab_size,
                                  embedding_dim=emb_dims,
                                  pretrained_embeddings=vocab_embeddings)

        logging.info(extractor)

        df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        df_val = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))

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

        if configs.reinforce.length_limit > 0:
            configs.reinforce.length_limit = 2

        device = "cpu"
        if args.cuda:
            device = "cuda"

        net = train_extractor(extractor=extractor,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              device=device,
                              configs=configs)

    else:
        df = None
        if args.eval_data == "test":
            df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        else:
            df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

        vocab = Vocab(vocab_file_path=args.vocab_file,
                      glove_file_path=args.glove_file,
                      embedding_size=args.embedding_size)
        vocab_embeddings = vocab.get_embedding()
        vocab_size, emb_dims = vocab_embeddings.shape

        eval_dataset = CNNDailyMailDataset(df=df,
                                           word2id=vocab)
        eval_dataloader = BatchDataLoader(eval_dataset,
                                          batch_size=1,
                                          shuffle=False)

        logging.info(f"Load model from {args.pretrained_model}")
        device = "cpu"
        if args.cuda:
            device = "cuda"
        try:
            extractor = torch.load(args.pretrained_model,
                                   map_location=lambda storage, loc: storage)
            extractor.to(device)
        except:
            extractor = get_extractor(configs.model,
                                      vocab_size=vocab_size,
                                      embedding_dim=emb_dims,
                                      pretrained_embeddings=vocab_embeddings)
            extractor.load_state_dict(torch.load(args.pretrained_model))
            extractor.to(device)
        avg_rouge, avg_lead3 = eval_extractor(extractor,
                                              configs=configs,
                                              dataloader=eval_dataloader,
                                              eval_data=args.eval_data,
                                              device=device)
        columns = ["rouge-1(p)", "rouge-1(r)", "rouge-1(f)",
                   "rouge-2(p)", "rouge-2(r)", "rouge-2(f)",
                   "rouge-l(p)", "rouge-l(r)", "rouge-l(f)"]
        logging.info(f"Avg_rouge: {avg_rouge}")
        logging.info(f"Avg_lead3: {avg_lead3}")
        s = pd.Series(['Avg_HER', "Avg_lead3"])
        # result = pd.DataFrame([avg_rouge, avg_lead3])
        # result.set_index(s)
        # result.to_csv("result_test_2.csv")
