import argparse

from modules.data.dataloader import CNNDailyMailDataset, BatchDataLoader
from modules.data.vocab import Vocab
from modules.models.simple_rnn import SimpleRNN
from modules.reinforce.reward import ReinforceReward
from modules.reinforce.rouge import reinforce_loss


from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import logging
from typing import Any
import pandas as pd


def eval_model(extractor: torch.nn.Module,
               dataloader: BatchDataLoader,
               eval_data: str = "test",
               std_rouge: bool = False,
               length_limit: int = -1,
               rouge_metric="all"):
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
                if args.oracle_length == -1:  # use true oracle length
                    oracle_summary_sent_num = len(doc.tokenized_summary)
                else:
                    oracle_summary_sent_num = args.oracle_length

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
                                                     std_rouge=std_rouge,
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


def summarizer_train(args):
    vocab = Vocab(vocab_file_path="data/vocab",
                  glove_file_path="data/vocab_100d.txt",
                  embedding_size=100)

    df_train = pd.read_csv("data/cnn_dailymail/train.csv")
    df_val = pd.read_csv("data/cnn_dailymail/valid.csv")

    df_train = df_train.iloc[:50]
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

    model_name = ".".join((args.model_file,
                           str(args.ext_model),
                           str(args.rouge_metric),
                           str(args.std_rouge),
                           str(args.rl_baseline_method),
                           "oracle_l", str(args.oracle_length),
                           "bsz", str(args.sample_size),
                           "rl_loss", str(args.rl_loss_method),
                           "train_example_quota", str(
                               args.train_example_quota),
                           "length_limit", str(args.length_limit),
                           "hidden", str(args.hidden),
                           "dropout", str(args.dropout),
                           'ext'))

    log_name = ".".join(("./logs/model",
                         str(args.ext_model),
                         str(args.rouge_metric), str(args.std_rouge),
                         str(args.rl_baseline_method), "oracle_l", str(
                             args.oracle_length),
                         "bsz", str(args.sample_size), "rl_loss", str(
                             args.rl_loss_method),
                         "train_example_quota", str(args.train_example_quota),
                         "length_limit", str(args.length_limit),
                         "hidden", str(args.hidden),
                         "dropout", str(args.dropout),
                         'ext'))

    vocab_embeddings = vocab.get_embedding()
    vocab_size, emb_dims = vocab_embeddings.shape
    extractor = SimpleRNN(
        vocab_size=vocab_size,
        embedding_dim=emb_dims,
        word_input_size=100,
        sentence_input_size=400,
        dropout_p=0,
        word_lstm_hidden_units=200,
        sentence_lstm_hidden_units=200,
        word_pretrained_embedding=vocab_embeddings
    )
    extractor = extractor.cuda()

    optimizer_ext = torch.optim.Adam(
        extractor.parameters(), lr=args.lr, betas=(0., 0.999))
    print("starting training")
    n_step = 100

    # init statistics
    reward_list = []
    best_eval_reward = 0.
    model_save_name = model_name
    logging.basicConfig(handlers=[logging.FileHandler('%s.log' % log_name), logging.StreamHandler()],
                        level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    reinforce = ReinforceReward(std_rouge=args.std_rouge, rouge_metric=args.rouge_metric,
                                sample_size=args.sample_size, rl_baseline_method=args.rl_baseline_method,
                                loss_method=1)
    messsage = '[Train] - Epoch %d Step %d Reward %.4f'
    for epoch in range(args.epochs_ext):
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
                    if args.oracle_length == -1:  # use true oracle length
                        oracle_summary_sent_num = len(doc.tokenized_summary)
                    else:
                        oracle_summary_sent_num = args.oracle_length

                    x = doc.tokenized_content
                    if min(x.shape) == 0:
                        continue
                    sents = Variable(torch.from_numpy(x)).cuda()
                    outputs = extractor(sents)

                    if args.prt_inf and np.random.randint(0, 100) == 0:
                        prt = True
                    else:
                        prt = False

                    loss, reward = reinforce.train(outputs,
                                                   doc,
                                                   max_num_of_sents=oracle_summary_sent_num,
                                                   max_num_of_bytes=args.length_limit,
                                                   prt=prt)
                    if prt:
                        print('Probabilities: ',
                              outputs.squeeze().data.cpu().numpy())
                        print('-' * 80)

                    reward_list.append(reward)

                    if isinstance(loss, Variable):
                        loss.backward()

                    if step % 1 == 0:
                        torch.nn.utils.clip_grad_norm(
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

            if len(train_dataset) == step_in_epoch or ((step_in_epoch) % 5000 == 0 and step_in_epoch != 0):
                eval_reward, lead3_reward = eval_model(
                    extractor, val_dataloader,
                    eval_data="val",
                    std_rouge=args.std_rouge,
                    length_limit=args.length_limit,
                    rouge_metric="avg_f")
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    logging.info("Saving model %s with eval_reward: %s leadreward: %s" %
                                 (model_save_name, eval_reward, lead3_reward))
                    torch.save(extractor, model_name)
                logging.info('[Eval] - Epoch ' + str(epoch) +
                             ' Mean reward: ' + str(np.mean(eval_reward)))

    return extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str,
                        default='./data/CNN_DM_pickle_data/vocab_100d.p')
    parser.add_argument('--data_dir', type=str,
                        default='./data/CNN_DM_pickle_data/')
    parser.add_argument('--model_file', type=str,
                        default='./model/summary.model')
    parser.add_argument('--epochs_ext', type=int, default=10)
    parser.add_argument('--load_ext', action='store_true')
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--std_rouge', type=bool)

    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rouge_metric', type=str, default='avg_f')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--sample_size', type=int, default=20)
    parser.add_argument('--fine_tune', action='store_true',
                        help='fine tune with std rouge')
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')
    parser.add_argument('--ext_model', type=str, default="simpleRNN",
                        help='lstm_summarunner, gru_summarunner, bag_of_words, simpleRNN')
    parser.add_argument('--prt_inf', action='store_true')

    args = parser.parse_args()

    if args.length_limit > 0:
        args.oracle_length = 2
    net = summarizer_train(args)
