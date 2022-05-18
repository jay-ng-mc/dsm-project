from river import compose
from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing
from river import tree

X_y = datasets.Taxis()

model = compose.Select('dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'pickup_latitude',
                       'pickup_longitude', )

model |= preprocessing.StandardScaler()
model += (compose.Select('store_and_fwd_flag', 'vendor_id') | preprocessing.OneHotEncoder())

if True:
    model |= tree.HoeffdingOptionTreeRegressor()
else:
    model |= tree.HoeffdingTreeRegressor()

metric = metrics.MAE()

evaluate.progressive_val_score(X_y, model, metric, print_every=1000)

