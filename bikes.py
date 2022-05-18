from river import compose
from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing
from river import tree

X_y = datasets.Bikes()

model = compose.Select('clouds', 'humidity', 'pressure', 'wind', 'temperature')

model |= preprocessing.StandardScaler()

model |= tree.HoeffdingOptionTreeRegressor()

metric = metrics.MAE()

evaluate.progressive_val_score(X_y, model, metric, print_every=1000)

