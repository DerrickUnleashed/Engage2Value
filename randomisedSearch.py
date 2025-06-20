import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from tqdm import tqdm
warnings.filterwarnings('ignore')

#Loading Dataset
df = pd.read_csv('train_data.csv')
len(list(df.columns))
pd.set_option('display.max_rows', None)

#Dropping Useless columns
fields_to_remove = [
    "device.screenResolution",
    "screenSize",
    "device.mobileDeviceBranding",
    "device.mobileInputSelector",
    "device.mobileDeviceMarketingName",
    "device.operatingSystemVersion",
    "device.flashVersion",
    "geoNetwork.networkLocation",
    "browserMajor",
    "device.browserSize",
    "socialEngagementType",
    "device.mobileDeviceModel",
    "device.language",
    "device.browserVersion",
    "device.screenColors",
    "totals.bounces",
    "locationZone",
    "totals.visits",
    "new_visits"
]
df = df.drop(columns = fields_to_remove,axis = 1)
len(list(df.columns))

non_numeric_cols = df.select_dtypes(exclude=['number'])
non_numeric_cols.columns


# ## Label Encoding with frequency
# - browser
# - trafficSource.adContent
# - trafficSource.campaign
# - geoNetwork.subContinent
# 
# ## Label Encoding
# - geoCluster
# - os
# - geoNetwork.networkDomain
# - trafficSource.isTrueDirect
# - trafficSource.adwordsClickInfo.slot
# - trafficSource.medium
# - trafficSource.adwordsClickInfo.isVideoAd
# - trafficSource.adwordsClickInfo.adNetworkType
# - deviceType
# - userChannel
# - geoNetwork.continent
# - device.isMobile
# 
# ## Text Vectorization (TF-IDF)
# - trafficSource.keyword
# - geoNetwork.region
# - locationCountry
# - geoNetwork.city
# - geoNetwork.metro
# - trafficSource.referralPath
# - trafficSource

#Label encoding
label_cols = [
    'geoCluster',
    'os',
    'geoNetwork.networkDomain',
    'trafficSource.isTrueDirect',
    'trafficSource.adwordsClickInfo.slot',
    'trafficSource.medium',
    'trafficSource.adwordsClickInfo.isVideoAd',
    'trafficSource.adwordsClickInfo.adNetworkType',
    'deviceType',
    'userChannel',
    'geoNetwork.continent',
    'device.isMobile'
]
le = LabelEncoder()
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

#Frequency Encoding
freq_label_cols = [
    'browser',
    'trafficSource.adContent',
    'trafficSource.campaign',
    'geoNetwork.subContinent'
]
freq_encodings = {}
for col in freq_label_cols:
    freq = df[col].value_counts()
    encoding = {k: i for i, k in enumerate(freq.index, start=1)}
    encoding = {k: len(encoding) - v + 1 for k, v in encoding.items()}  # reverse rank
    freq_encodings[col] = encoding
    df[col] = df[col].map(encoding).fillna(0).astype(int)  # unknowns as 0

#Text Vectorization
text_cols = [
    'trafficSource.keyword',
    'geoNetwork.region',
    'locationCountry',
    'geoNetwork.city',
    'geoNetwork.metro',
    'trafficSource.referralPath',
    'trafficSource'
]
tfidf_encoders = {}
tfidf_feature_parts = []
for col in text_cols:
    tfidf = TfidfVectorizer(max_features=10, stop_words='english')  # you can tune max_features
    transformed = tfidf.fit_transform(df[col].fillna(''))
    tfidf_df = pd.DataFrame(
        transformed.toarray(),
        columns=[f"{col}_tfidf_{i}" for i in range(transformed.shape[1])]
    )
    df = df.drop(columns=col).reset_index(drop=True)
    df = pd.concat([df, tfidf_df], axis=1)
    tfidf_encoders[col] = tfidf
    tfidf_feature_parts.extend(tfidf_df.columns)

with open("tfidf_encoders.pkl", "wb") as f:
    pickle.dump(tfidf_encoders, f)

#Removing Nulls
pd.set_option('display.max_rows', None)
df['trafficSource.adwordsClickInfo.page'].fillna(0.0,inplace = True)
df['pageViews'].fillna(0.0,inplace = True)

#Just Checking
df.isnull().sum().sum()

#Heat Map
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
threshold = 0.95
strong_corr = corr_matrix.where(mask).stack().reset_index()
strong_corr.columns = ['Feature1', 'Feature2', 'Correlation']
strong_corr = strong_corr[strong_corr['Correlation'].abs() >= threshold]
filtered_features = pd.unique(strong_corr[['Feature1', 'Feature2']].values.ravel())
filtered_corr_matrix = df[filtered_features].corr()
# plt.figure(figsize=(100, 80))
# sns.heatmap(filtered_corr_matrix, annot=True, cmap='YlGnBu')
# plt.title("Filtered Correlation Heatmap (|corr| ≥ 0.95)")
# plt.tight_layout()
# plt.show()

#Extreme correlation check
for _, row in strong_corr.iterrows():
    f1, f2, corr = row['Feature1'], row['Feature2'], row['Correlation']
    if f1 != f2:
        print(f"{f1}  <-->  {f2}  |  Corr: {corr:.3f}")

#Dropping High corr columns
df.drop(columns = ['sessionId'],axis = 1, inplace = True)

# Define features and target
X = df.drop(columns='purchaseValue')  # Replace 'target' with your actual target column name
y = df['purchaseValue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter distributions
param_dist = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': randint(0, 6),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.1, 10)
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=150,            # Tune this up or down
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit search
print("🔍 Starting RandomizedSearchCV (scoring = R²)...")
random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_
print("\n✅ Best Parameters Found:")
print(random_search.best_params_)

# Predict
y_pred = best_xgb.predict(X_test)

# Evaluation
print("\n📊 Evaluation on Test Set:")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred, squared=False))

# Save model
with open('best_xgb_random_search.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

print("\n💾 Model saved as 'best_xgb_random_search.pkl'")
