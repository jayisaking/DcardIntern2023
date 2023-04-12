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
    parser = argparse.ArgumentParser('Set parameters for training Likes Prediction Model', add_help=False)
    parser.add_argument('--lr_regressor', default = 1e-4, type = float)
    parser.add_argument('--lr_bert', default = 5e-5, type = float)
    parser.add_argument('--checkpoint_dir', default = './', help = 'for storing models')
    parser.add_argument('--batch_size', default = 128, type = int)
    parser.add_argument('--epochs', default = 256, type = int)
    parser.add_argument('--save_every', default = 5, type = int)
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--nn_checkpoint', default = None, help = 'checkpoint for neural net')
    parser.add_argument('--train_data', default = None)
    parser.add_argument('--validation_data', default = None)
    parser.add_argument('--bert_type', default = 'hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--embed_dim', default = 128)
    parser.add_argument('--xgb_estimators', default = 1526)
    parser.add_argument('--xgb_lr', default = 1e-2)
    parser.add_argument('--xgb_colsample_bytree', default = 0.9)
    parser.add_argument('--xgb_max_depth', default = None)
    parser.add_argument('--plot_output_dir', default = './')
    args = parser.parse_args()
    return args
def main(args):
    """
    This function trains a regression model using a combination of a BERT model and an XGBoost
    regressor, and saves the resulting MAPE plot.
    
    :param args: args is a command line argument parser object that contains various arguments passed to
    the script at runtime. These arguments include paths to input data files, hyperparameters for the
    model, and output directories for saving results
    """
    train_data = pd.read_csv(args.train_data)
    validation_data = pd.read_csv(args.validation_data)
    bert = BertModel.from_pretrained(args.bert_type)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_type)
    device = torch.device(args.device)
    train_dataset = LikesDataset(train_data, tokenizer, device = device)
    validation_dataset = LikesDataset(validation_data, tokenizer, device = device)
    model = LikesRegression(bert = bert, tokenizer = tokenizer, train_dataset = train_dataset, test_dataset = validation_dataset, batch_size = args.batch_size, lr = args.lr_regressor,
                            lr_bert = args.lr_bert, embed_dim = args.embed_dim)
    model.to(device)
    if args.nn_checkpoint is not None:
        if isinstance(torch.load(args.nn_checkpoint), LikesRegression):
            model.load_state_dict(torch.load(args.nn_checkpoint).state_dict())
        else:
            model.load_state_dict(torch.load(args.nn_checkpoint))
    xgb = XGBRegressor(
    random_state = 100,
    n_estimators = args.xgb_estimators,
    max_depth = args.xgb_max_depth,
    learning_rate = args.xgb_lr,
    colsample_bytree = args.xgb_colsample_bytree)
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()
    optimizer = model.configure_optimizers()
    val_mse = []
    val_mae = []
    val_mape = []
    for epoch in range(args.epochs):
        (temp_mse, temp_mae, temp_mape), xgb, model = run_one_epoch(train_dataloader, val_dataloader, epoch, model, device, optimizer, xgb, args.save_every, args.checkpoint_dir)
        val_mse.extend(temp_mse)
        val_mae.extend(temp_mae)
        val_mape.extend(temp_mape)
    plt.plot(val_mape)
    plt.xlabel('steps')
    plt.ylabel('MAPE')
    plt.savefig(os.path.join(args.plot_output_dir, 'mape.png'))
if __name__ == '__main__':
    main(get_args_parser())
    