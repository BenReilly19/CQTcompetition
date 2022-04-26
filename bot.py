from ib_insync import *
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
import math
from sklearn.metrics import mean_squared_error


def bot(stock):
    # connect to IB instance
    ib = IB()
    ib.connect()

    contract = Stock(stock, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # Download historical data for stock up to current price
    print("Downloading Historical data")
    hist = ib.reqHistoricalData(
        contract,
        '',
        barSizeSetting='15 mins',
        durationStr='2 Y',
        whatToShow='MIDPOINT',
        useRTH=True
    )

    df = util.df(hist)
    price = df[['close']]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['close'] = scaler.fit_transform(price['close'].values.reshape(-1, 1))

    def split_data(stock, lookback):
        data_raw = stock.to_numpy()  # convert to numpy array
        data = []

    # create all possible sequences of length seq_len
        for index in range(len(data_raw) - lookback):
            data.append(data_raw[index: index + lookback])

        data = np.array(data)
        test_set_size = int(np.round(0.2*data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        return [x_train, y_train, x_test, y_test]

    lookback = 20  # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)

    x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.gru = nn.GRU(input_dim, hidden_dim,
                              num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(
                0), self.hidden_dim).requires_grad_()
            out, (hn) = self.gru(x, (h0.detach()))
            out = self.fc(out[:, -1, :])
            return out

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    print('Training model')
    for t in range(num_epochs):
        y_train_pred = model(x_train_tensor)

        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))

    y_test_pred = model(x_test_tensor)

    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

    trainScore = math.sqrt(mean_squared_error(
        y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

    while True:

        # Once trained, predict next price
        current_data = util.df(ib.reqHistoricalData(
            contract,
            '',
            barSizeSetting='15 mins',
            durationStr='1 D',
            whatToShow='MIDPOINT',
            useRTH=True
        ))

        current_pricedata = current_data[['close']]

        current_pricedata['close'] = scaler.fit_transform(
            current_pricedata['close'].values.reshape(-1, 1))

        def train_clean(stock, lookback):
            data_raw = stock.to_numpy()  # convert to numpy array
            data = []

        # create all possible sequences of length seq_len
            for index in range(len(data_raw) - lookback):
                data.append(data_raw[index: index + lookback])

            data = np.array(data)
            test_set_size = 0
            train_set_size = data.shape[0] - (test_set_size)

            x_train = data[:train_set_size, :-1, :]
            y_train = data[:train_set_size, -1, :]

            return [x_train, y_train]

        lookback = 20  # choose sequence length
        xTe, yTe = train_clean(price, lookback)

        xTe_Tensor = torch.from_numpy(xTe).type(torch.Tensor)

        pred = model(xTe_Tensor)
        pred_df = pd.DataFrame(scaler.inverse_transform(pred.detach().numpy()))
        pred_price = float(pred_df.loc[0])
        print("Predicted price next 15 minutes: ", pred_price)

        # sell all stonks

        positions = ib.positions()
        for i in range(len(positions)):
            if positions[i].contract.symbol == stock:
                print(f"Selling {positions[i].position} of {stock}")
                ib.placeOrder(contract, MarketOrder(
                    'SELL', positions[i].position))

        # if predicted price > most current price: buy stock
        if pred_price > float(current_data['close'].loc[len(current_data) - 1]):
            print(f"Buying 10 of {stock}")
            ib.placeOrder(contract, MarketOrder('BUY', 3000))

            # sleep for 15 min
        ib.sleep(900)
        # repeat


if __name__ == "__main__":
    bot('AAPL')
