from pprint import pprint
from river import datasets

X_y = datasets.Bikes()

for x, y in X_y:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break

from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim
from river import tree

X_y = datasets.Bikes()

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
# model = compose.Pipeline()
model |= preprocessing.StandardScaler()
# model |= tree.HoeffdingAdaptiveTreeRegressor()
# model |= tree.HoeffdingTreeRegressor()
model |= tree.HoeffdingOptionTreeRegressor()

metric = metrics.MAE()

evaluate.progressive_val_score(X_y, model, metric, print_every=20_000)

