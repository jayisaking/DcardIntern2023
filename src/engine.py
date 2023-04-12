from modules import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
def xgb_result(xgb: XGBRegressor, x_train, y_train, x_test, y_test):
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(np.array(x_test)).reshape(-1)
    print("XGB MSE:", mean_squared_error(y_pred, y_test))
    print("XGB MAE:", mean_absolute_error(y_pred, y_test))
    print("XGB MAPE:", mean_absolute_percentage_error(y_pred, y_test))
    return xgb
def run_one_epoch(train_dataloader: DataLoader, val_dataloader: DataLoader, epoch, model: LikesRegression, device, optimizer, xgb: XGBRegressor, save_every, checkpoint_dir):
    val_mse = []
    val_mae = []
    val_mape = []
        # The `for idx, batch in enumerate(train_dataloader):` loop iterates over each batch in the training dataloader, and
        # the `t.update()` line updates the progress bar to reflect the completion of each batch. The
        # `t.set_postfix(mse = mse.item(), mae = mae.item(), mape = mape.item())` line updates the progress
        # bar to include the current values of the mean squared error (MSE), mean absolute error (MAE), and
        # mean absolute percentage error (MAPE) for the current batch. Finally, the `if idx % 500 == 0 and
        # idx != 0:` block saves the current state of the neural network model and the XGBoost model every
        # 500 batches.
    with tqdm(total = len(train_dataloader)) as t:
        t.set_description('Epoch %i' % epoch)
        xgb_x_train = []
        xgb_y_train = []
        xgb_x_val = []
        xgb_y_val = []
        for idx, batch in enumerate(train_dataloader):
            mse, mae, mape, for_tree = model.step(batch, idx, device)
            xgb_x_train.append(np.concatenate((for_tree[0].reshape((len(for_tree[1]), -1)), for_tree[1], for_tree[2]), axis = 1))
            xgb_y_train.append(for_tree[3])
            optimizer.zero_grad()
            # mse.backward()
            # mae.backward()
            mape.backward()
            optimizer.step()
            t.set_postfix(mse = mse.item(), mae = mae.item(), mape = mape.item())
            t.update()
            if idx % 500 == 0 and idx != 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"nn_latest.pth"))
                if epoch > save_every:
                    xgb.save_model(os.path.join(checkpoint_dir, f"xgb_latest.txt"))  
        # This code block is evaluating the performance of the model on the validation dataset for a given
        # epoch. The `torch.no_grad()` context manager is used to disable gradient computation, which reduces
        # memory usage and speeds up computation. The `model.step()` function is called on each batch of the validation dataset to
        # obtain the model's predictions and compute the mean squared error (MSE), mean absolute error (MAE),
        # and mean absolute percentage error (MAPE) between the predictions and the ground truth labels. These
        # metrics are then appended to lists `val_mse`, `val_mae`, and `val_mape`. Finally, the average of
        # these metrics is printed as the performance of the model on the validation dataset for the given
        # epoch.
    with torch.no_grad():
        with tqdm(total = len(val_dataloader)) as t:
            t.set_description('Epoch %i: validation' % epoch)
            for idx, batch in enumerate(val_dataloader):
                mse, mae, mape, for_tree = model.step(batch, idx, device)
                xgb_x_val.append(np.concatenate((for_tree[0].reshape((len(for_tree[1]), -1)), for_tree[1], for_tree[2]), axis = 1))
                xgb_y_val.append(for_tree[3])
                t.set_postfix(mse = mse.item(), mae = mae.item(), mape = mape.item())
                t.update()
                val_mse.append(mse.item())
                val_mae.append(mae.item())
                val_mape.append(mape.item())
        print("NET MSE:", np.mean(val_mse), "MAE:", np.mean(val_mae), "MAPE:", np.mean(val_mape))
    if epoch % save_every == 0 and epoch != 0:
        xgb_result(xgb = xgb, x_train = np.concatenate(xgb_x_train, axis = 0), y_train = np.concatenate(xgb_y_train, axis = 0), x_test = np.concatenate(xgb_x_val, axis = 0), y_test = np.concatenate(xgb_y_val, axis = 0))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"nn_epoch{epoch}.pth"))
        xgb.save_model(os.path.join(checkpoint_dir, f"xgb_epoch{epoch}.txt"))     
    return (val_mse, val_mae, val_mape), xgb, model

    