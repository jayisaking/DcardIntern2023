from xgboost import XGBRegressor
import os
import argparse
import pandas as pd
from transformers import BertModel, BertTokenizerFast
from modules import *
import torch
import matplotlib.pyplot as plt
from engine import run_one_epoch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for inferencing Likes Prediction Model', add_help=False)
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--nn_checkpoint', default = None, help = 'checkpoint for neural net')
    parser.add_argument('--xgb_checkpoint', default = None, help = 'cehckpoint for XGBoost')
    parser.add_argument('--test_data', default = None)
    parser.add_argument('--submit_example_file', default = './data/example_result.csv')
    parser.add_argument('--submit_file', default = './submit.csv')
    parser.add_argument('--bert_type', default = 'hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--embed_dim', default = 128)
    parser.add_argument('--xgb_estimators', default = 1526)
    parser.add_argument('--xgb_lr', default = 1e-2)
    parser.add_argument('--xgb_colsample_bytree', default = 0.9)
    parser.add_argument('--xgb_max_depth', default = None)
    args = parser.parse_args()
    return args
@torch.no_grad()
def main(args):
    test_data = pd.read_csv(args.test_data)
    bert = BertModel.from_pretrained(args.bert_type)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_type)
    device = torch.device(args.device)
    model = LikesRegression(bert = bert, tokenizer = tokenizer, addition_dim = 15)
    model.to(device)
    if args.nn_checkpoint is not None:
        if isinstance(torch.load(args.nn_checkpoint), LikesRegression):
            model.load_state_dict(torch.load(args.nn_checkpoint).state_dict())
        else:
            model.load_state_dict(torch.load(args.nn_checkpoint))
    model.eval()
    submit = pd.read_csv(args.submit_example_file)
    xgb = XGBRegressor(
    random_state = 100,
    n_estimators = args.xgb_estimators,
    max_depth = args.xgb_max_depth,
    learning_rate = args.xgb_lr,
    colsample_bytree = args.xgb_colsample_bytree)
    xgb.load_model(args.xgb_checkpoint)
    net_results = []
    for row in test_data.to_numpy():
        counts = torch.tensor(row[2: -3].astype(np.float64), device = device).float()
        times = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S UTC')
        title = tokenizer.encode(row[0], return_tensors = 'pt').to(device)
        addition = torch.cat((counts.to(device), torch.tensor([times.hour, times.minute, times.second], device = device)), dim = 0).unsqueeze(0)
        value = model(title, addition.to(device), mask = torch.ones((1, len(title))).to(device))
        net_results.append([value.item()].extend(addition.detach().clone().view(-1).tolist()).extend([times.hour, times.minute, times.second]).extend(row[-3:].tolist()))
    net_results = np.array(net_results)
    results = np.round(xgb.predict(net_results)).astype(np.int64)
    submit['like_count_24h'] = results
    submit.to_csv(args.submit_file)
if __name__ == '__main__':
    main(get_args_parser())