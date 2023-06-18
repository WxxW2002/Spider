import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from tqdm import tqdm
import pickle
import sys

if __name__ == '__main__':
    df = pd.read_csv('./data/house_processed.csv')
    # df = pd.read_csv('./data/house_with_embeddings.csv')

    X = df.drop(['Title', 'Subtitle', 'Total', 'Average'], axis=1)
    y = df['Average']
    y = y / 10000 # unit of measure
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    parser = argparse.ArgumentParser(description='Predict house average price.')
    parser.add_argument('--model', type=str, choices=['linear', 'decision_tree', 'random_forest', 'adaboost', 'bagging', 'gradient_boost', 'svr', 'bayesian_ridge', 'elastic_net', 'mlp'], default='linear',
                        help='Model to use for prediction')
    parser.add_argument('--hidden_layer_sizes', type=str, default='(100,)', help='Hidden layer sizes for MLPRegressor')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for MLPRegressor')
    args = parser.parse_args()

    if args.model == 'linear':
        model = LinearRegression()
    elif args.model == 'decision_tree':
        model = DecisionTreeRegressor()
    elif args.model == 'random_forest':
        model = RandomForestRegressor()
    elif args.model == 'adaboost':
        model = AdaBoostRegressor()
    elif args.model == 'bagging':
        model = BaggingRegressor()
    elif args.model == 'gradient_boost':
        model = GradientBoostingRegressor()
    elif args.model == 'svr':
        model = SVR()
    elif args.model == 'bayesian_ridge':
        model = BayesianRidge()
    elif args.model == 'elastic_net':
        model = ElasticNet()
    elif args.model == 'mlp':
        hidden_layer_sizes = eval(args.hidden_layer_sizes)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=args.max_iter)

    print('Training...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    results.to_csv(f'out/prediction/{args.model}.csv', index=False)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    print('均方误差 (MSE):', mse)
    print('平均绝对误差 (MAE):', mae)
    print('决定系数 (R^2):', r2)
    print('调整的决定系数 (Adjusted R^2):', adjusted_r2)

    output_file = f'out/statistics/{args.model}.txt'
    with open(output_file, 'w') as f:
        f.write(f'run: python {" ".join(sys.argv)}\n')
        f.write(f'均方误差 (MSE): {mse}\n')
        f.write(f'平均绝对误差 (MAE): {mae}\n')
        f.write(f'决定系数 (R^2): {r2}\n')
        f.write(f'调整的决定系数 (Adjusted R^2): {adjusted_r2}\n')

    # with open(f'./out/models/{args.model}.pkl', 'wb') as f:
    #     pickle.dump(model, f)
