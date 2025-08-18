import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# === Load Data ===
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
transactions = pd.read_csv('transactions.csv')

try:
    holidays = pd.read_csv('holidays.csv')
    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays['importance'] = holidays['type'].map({
        'major': 2, 'festival': 1.5, 'minor': 1, 'optional': 0.5
    }).fillna(1.0)
    has_holidays = True
except:
    has_holidays = False

# === Convert Dates ===
for df in [train, test, transactions]:
    df['doj'] = pd.to_datetime(df['doj'])
transactions['doi'] = pd.to_datetime(transactions['doi'])
transactions = transactions.sort_values(by=['srcid', 'destid', 'doj'])

# === Lag Features ===
for window in [3, 7]:
    transactions[f'rolling_seatcount_mean_{window}'] = (
        transactions.groupby(['srcid', 'destid'])['cumsum_seatcount']
        .transform(lambda x: x.shift(1).rolling(window=window).mean())
    )
    transactions[f'rolling_searchcount_mean_{window}'] = (
        transactions.groupby(['srcid', 'destid'])['cumsum_searchcount']
        .transform(lambda x: x.shift(1).rolling(window=window).mean())
    )
transactions.fillna(0, inplace=True)

# Filter and Aggregate
tx_15 = transactions[transactions['dbd'] == 15].copy()
route_agg = tx_15.groupby(['srcid', 'destid']).agg({
    'cumsum_seatcount': ['mean', 'std'],
    'cumsum_searchcount': ['mean', 'std']
})
route_agg.columns = ['seatcount_mean', 'seatcount_std', 'searchcount_mean', 'searchcount_std']
route_agg.reset_index(inplace=True)

# Merge
train_merged = pd.merge(train, tx_15, on=['doj', 'srcid', 'destid'], how='inner')
test_merged = pd.merge(test, tx_15, on=['doj', 'srcid', 'destid'], how='left')
train_merged = pd.merge(train_merged, route_agg, on=['srcid', 'destid'], how='left')
test_merged = pd.merge(test_merged, route_agg, on=['srcid', 'destid'], how='left')

# === Feature Engineering ===
def add_features(df):
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
    df['month'] = df['doj'].dt.month
    df['search_seat_ratio'] = df['cumsum_searchcount'] / (df['cumsum_seatcount'] + 1)
    if has_holidays:
        df = df.merge(holidays[['date', 'importance']], left_on='doj', right_on='date', how='left')
        df['holiday_importance'] = df['importance'].fillna(0)
        df.drop(['date', 'importance'], axis=1, inplace=True)
    else:
        df['holiday_importance'] = 0
    for col in ['srcid_tier', 'destid_tier']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype('category').cat.codes
    return df

train_merged = add_features(train_merged)
test_merged = add_features(test_merged)

# Fill Missing
for col in ['seatcount_mean', 'seatcount_std', 'searchcount_mean', 'searchcount_std']:
    test_merged[col].fillna(train_merged[col].mean(), inplace=True)
test_merged.fillna(0, inplace=True)

# === Features and Target ===
features = [
    'cumsum_seatcount', 'cumsum_searchcount', 'seatcount_mean', 'seatcount_std',
    'searchcount_mean', 'searchcount_std', 'search_seat_ratio',
    'rolling_seatcount_mean_3', 'rolling_searchcount_mean_3',
    'rolling_seatcount_mean_7', 'rolling_searchcount_mean_7',
    'srcid_tier', 'destid_tier', 'day_of_week', 'is_weekend',
    'week_of_year', 'month', 'holiday_importance'
]

target = 'final_seatcount'
X = train_merged[features]
y = train_merged[target].astype(np.float32)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# === Models ===
lgb_model = lgb.LGBMRegressor(
    n_estimators=1200, learning_rate=0.01, max_depth=7, num_leaves=50,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.015, max_depth=6, subsample=0.85,
    colsample_bytree=0.85, random_state=42
)

# Train base models
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Validation Predictions
val_preds_lgb = lgb_model.predict(X_valid)
val_preds_xgb = xgb_model.predict(X_valid)
meta_X = np.vstack([val_preds_lgb, val_preds_xgb]).T

# Train meta-model (stacking)
meta_model = Ridge()
meta_model.fit(meta_X, y_valid)

# Report validation RMSE
stacked_val_preds = meta_model.predict(meta_X)
rmse = np.sqrt(mean_squared_error(y_valid, stacked_val_preds))
print(f"[Stacked Ridge Ensemble] RMSE: {rmse:.4f}")

# === Predict on Test ===
lgb_model.fit(X, y)
xgb_model.fit(X, y)

preds_lgb = lgb_model.predict(test_merged[features])
preds_xgb = xgb_model.predict(test_merged[features])
meta_test = np.vstack([preds_lgb, preds_xgb]).T
final_preds = np.round(meta_model.predict(meta_test)).astype(int)
final_preds = np.maximum(0, final_preds)

submission = pd.DataFrame({
    'route_key': test_merged['route_key'],
    'final_seatcount': final_preds
})
submission.to_csv('xgb_lgbm_stacking_final.csv', index=False)
print("\n xgb_lgbm_stacking_final.csv generated successfully!")

