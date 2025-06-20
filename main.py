import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(train_path, test_path=None):
    """
    Load and prepare the training and test datasets
    """
    print("Loading data...")
    
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Load test data if provided
    if test_path:
        test_df = pd.read_csv(test_path)
        # Combine for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    else:
        combined_df = train_df.copy()
    
    # Handle different date formats
    if 'date' in combined_df.columns:
        # Try different date parsing methods
        try:
            # First try standard datetime parsing
            combined_df['date'] = pd.to_datetime(combined_df['date'])
        except:
            try:
                # Try parsing as integer (Unix timestamp)
                combined_df['date'] = pd.to_datetime(combined_df['date'], unit='s')
            except:
                try:
                    # Try parsing as string with specific format
                    combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y%m%d')
                except:
                    print("Warning: Could not parse date column. Using sessionStart instead.")
                    if 'sessionStart' in combined_df.columns:
                        combined_df['date'] = pd.to_datetime(combined_df['sessionStart'], unit='s', errors='coerce')
                    else:
                        # Create dummy dates if all else fails
                        combined_df['date'] = pd.date_range(start='2018-01-01', periods=len(combined_df), freq='H')
    else:
        # If no date column, try sessionStart
        if 'sessionStart' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['sessionStart'], unit='s', errors='coerce')
        else:
            # Create dummy dates
            combined_df['date'] = pd.date_range(start='2018-01-01', periods=len(combined_df), freq='H')
    
    print(f"Date range after parsing: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Handle missing values in purchaseValue (0 means no purchase)
    combined_df['purchaseValue'] = combined_df['purchaseValue'].fillna(0)
    
    # Convert relevant columns to numeric where needed
    numeric_cols = ['sessionNumber', 'totals.visits', 'pageViews', 'totals.bounces', 
                   'totalHits', 'new_visits', 'purchaseValue']
    
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
    
    # Convert boolean columns
    bool_cols = ['trafficSource.isTrueDirect', 'gclIdPresent', 'device.isMobile', 
                'trafficSource.adwordsClickInfo.isVideoAd']
    
    for col in bool_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).map({'True': 1, 'False': 0, 'true': 1, 'false': 0}).fillna(0)
    
    return combined_df

def get_time_frame_with_features(data, window_num, window_days=168, gap_days=46, target_days=62):
    """
    Create time-based features similar to the R code logic
    
    Parameters:
    - data: Input dataframe
    - window_num: Which time window (1, 2, 3, etc.)
    - window_days: Days in each training window (168 = 24 weeks)
    - gap_days: Gap between training and target period (46 days)
    - target_days: Days in target period (62 days)
    """
    
    min_date = data['date'].min()
    
    # Define time windows
    window_start = min_date + timedelta(days=window_days * (window_num - 1))
    window_end = min_date + timedelta(days=window_days * window_num)
    
    target_start = window_end + timedelta(days=gap_days)
    target_end = target_start + timedelta(days=target_days)
    
    # Get data for the current window
    window_data = data[(data['date'] >= window_start) & (data['date'] < window_end)].copy()
    
    # Get users who return in the target period
    target_users = set(data[(data['date'] >= target_start) & (data['date'] < target_end)]['userId'].unique())
    
    # Get target data for returned users
    target_data = data[(data['userId'].isin(target_users)) & 
                      (data['date'] >= target_start) & 
                      (data['date'] < target_end)]
    
    # Calculate target revenue (log-transformed as in R code)
    target_revenue = target_data.groupby('userId')['purchaseValue'].sum().reset_index()
    target_revenue['target'] = np.log1p(target_revenue['purchaseValue'])  # log(1 + sum(revenue))
    target_revenue['returned'] = 1
    
    # Create features from window data
    window_features = create_user_features(window_data, window_start, window_end)
    
    # Add target information
    window_features = window_features.merge(
        target_revenue[['userId', 'target', 'returned']], 
        on='userId', 
        how='left'
    )
    
    # Fill NaN values for non-returned users
    window_features['target'] = window_features['target'].fillna(0)
    window_features['returned'] = window_features['returned'].fillna(0)
    
    return window_features

def create_user_features(data, period_start, period_end):
    """
    Create user-level features from session data
    """
    
    # Calculate time-based features
    data['days_from_start'] = (data['date'] - period_start).dt.days
    data['days_from_end'] = (period_end - data['date']).dt.days
    
    # Group by user and create features
    user_features = data.groupby('userId').agg({
        # Time-based features
        'days_from_start': ['min', 'max'],
        'days_from_end': ['min', 'max'],
        'date': ['nunique'],  # unique dates
        
        # Categorical features (take most frequent)
        'userChannel': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'browser': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'os': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'deviceType': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'locationCountry': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'geoNetwork.continent': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'geoNetwork.subContinent': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'geoNetwork.region': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'geoNetwork.city': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'trafficSource': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'trafficSource.medium': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        
        # Numerical aggregations
        'sessionNumber': 'max',
        'totalHits': ['sum', 'mean', 'min', 'max', 'std'],
        'pageViews': ['sum', 'mean', 'min', 'max', 'std'],
        'totals.visits': 'sum',
        'totals.bounces': 'sum',
        'new_visits': 'sum',
        'purchaseValue': 'sum',
        
        # Boolean features
        'device.isMobile': 'mean',
        'trafficSource.isTrueDirect': 'mean',
        'gclIdPresent': 'mean',
        'trafficSource.adwordsClickInfo.isVideoAd': 'mean',
        
        # Session count
        'sessionId': 'count'
    }).reset_index()
    
    # Flatten column names
    user_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in user_features.columns.values]
    
    # Handle missing values in std columns
    std_cols = [col for col in user_features.columns if col.endswith('_std')]
    for col in std_cols:
        user_features[col] = user_features[col].fillna(0)
    
    return user_features

def prepare_test_features(data, test_start_date='2018-05-01'):
    """
    Prepare features for test data (similar to tr5 in R code)
    """
    test_data = data[data['date'] >= test_start_date].copy()
    
    if test_data.empty:
        # If no specific test start date, use the last portion of data
        test_start_date = data['date'].quantile(0.8)
        test_data = data[data['date'] >= test_start_date].copy()
    
    period_start = test_data['date'].min()
    period_end = test_data['date'].max()
    
    test_features = create_user_features(test_data, period_start, period_end)
    test_features['target'] = np.nan
    test_features['returned'] = np.nan
    
    return test_features

def encode_categorical_features(train_data, test_data):
    """
    Encode categorical features consistently across train and test
    """
    categorical_cols = [col for col in train_data.columns 
                       if train_data[col].dtype == 'object' and col not in ['userId']]
    
    label_encoders = {}
    
    # Combine train and test for consistent encoding
    for col in categorical_cols:
        if col in train_data.columns and col in test_data.columns:
            le = LabelEncoder()
            
            # Combine unique values from both datasets
            combined_values = pd.concat([
                train_data[col].astype(str), 
                test_data[col].astype(str)
            ]).unique()
            
            le.fit(combined_values)
            
            train_data[col] = le.transform(train_data[col].astype(str))
            test_data[col] = le.transform(test_data[col].astype(str))
            
            label_encoders[col] = le
    
    return train_data, test_data, label_encoders

def train_models(train_data, n_iterations=10):
    """
    Train ensemble of classification and regression models
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Prepare features and targets
    feature_cols = [col for col in train_data.columns 
                   if col not in ['userId', 'target', 'returned']]
    
    X = train_data[feature_cols].values
    y_classification = train_data['returned'].values
    y_regression = train_data[train_data['returned'] == 1]['target'].values
    X_regression = train_data[train_data['returned'] == 1][feature_cols].values
    
    # Model parameters optimized for R² metric
    params_classification = {
        'objective': 'binary',
        'max_bin': 256,
        'learning_rate': 0.01,
        'num_leaves': 15,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 1,
        'bagging_freq': 1,
        'metric': 'binary_logloss',
        'verbose': -1,
        'random_state': 42
    }
    
    params_regression = {
        'objective': 'regression',
        'max_bin': 256,
        'learning_rate': 0.01,
        'num_leaves': 9,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 1,
        'bagging_freq': 1,
        'metric': 'rmse',  # RMSE correlates well with R²
        'verbose': -1,
        'random_state': 42
    }
    
    models_classification = []
    models_regression = []
    
    print("Training models...")
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")
        
        # Set different random states for each iteration
        params_classification['random_state'] = 42 + i
        params_regression['random_state'] = 42 + i
        
        # Train classification model with validation split
        if len(X) > 100:  # Only split if we have enough data
            X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
                X, y_classification, test_size=0.2, random_state=42+i, stratify=y_classification
            )
            
            train_data_clf = lgb.Dataset(X_train_clf, label=y_train_clf)
            val_data_clf = lgb.Dataset(X_val_clf, label=y_val_clf, reference=train_data_clf)
            
            model_clf = lgb.train(
                params_classification,
                train_data_clf,
                valid_sets=[val_data_clf],
                num_boost_round=1200,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            # If not enough data for validation split, train without early stopping
            train_data_clf = lgb.Dataset(X, label=y_classification)
            model_clf = lgb.train(
                params_classification,
                train_data_clf,
                num_boost_round=500,  # Reduced to prevent overfitting
                callbacks=[lgb.log_evaluation(0)]
            )
        
        models_classification.append(model_clf)
        
        # Train regression model with validation split
        if len(X_regression) > 100:  # Only split if we have enough data
            X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
                X_regression, y_regression, test_size=0.2, random_state=42+i
            )
            
            train_data_reg = lgb.Dataset(X_train_reg, label=y_train_reg)
            val_data_reg = lgb.Dataset(X_val_reg, label=y_val_reg, reference=train_data_reg)
            
            model_reg = lgb.train(
                params_regression,
                train_data_reg,
                valid_sets=[val_data_reg],
                num_boost_round=368,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            # If not enough data for validation split, train without early stopping
            train_data_reg = lgb.Dataset(X_regression, label=y_regression)
            model_reg = lgb.train(
                params_regression,
                train_data_reg,
                num_boost_round=200,  # Reduced to prevent overfitting
                callbacks=[lgb.log_evaluation(0)]
            )
        
        models_regression.append(model_reg)
    
    return models_classification, models_regression, feature_cols

def make_predictions(models_classification, models_regression, test_data, feature_cols):
    """
    Make predictions using ensemble of models
    """
    X_test = test_data[feature_cols].values
    
    predictions_sum = np.zeros(len(test_data))
    
    print("Making predictions...")
    for i, (model_clf, model_reg) in enumerate(zip(models_classification, models_regression)):
        # Predict probability of return
        prob_return = model_clf.predict(X_test)
        
        # Predict revenue for returners
        pred_revenue = model_reg.predict(X_test)
        
        # Combine predictions
        combined_pred = prob_return * pred_revenue
        predictions_sum += combined_pred
    
    # Average predictions
    final_predictions = predictions_sum / len(models_classification)
    
    return final_predictions

def evaluate_r2_score(y_true, y_pred):
    """
    Calculate R² score
    """
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def main():
    """
    Main execution function
    """
    # Load and prepare data
    # Replace with your actual file paths
    data = load_and_prepare_data('train_data.csv', 'test_data.csv')
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Unique users: {data['userId'].nunique()}")
    print(f"Users with purchases: {(data['purchaseValue'] > 0).sum()}")
    print(f"Total purchase value: {data['purchaseValue'].sum():.2f}")
    
    # Check if we have a reasonable date range for time windows
    date_range_days = (data['date'].max() - data['date'].min()).days
    print(f"Date range in days: {date_range_days}")
    
    if date_range_days > 500:  # If we have enough days for time windows
        print("Using time-window approach...")
        # Create time-based training sets
        train_parts = []
        
        # Create 4 training periods (similar to R code)
        for i in range(1, 5):
            print(f"Processing time window {i}")
            try:
                train_part = get_time_frame_with_features(data, i)
                if not train_part.empty and train_part['returned'].sum() > 0:
                    train_parts.append(train_part)
                    print(f"  Window {i}: {len(train_part)} users, {train_part['returned'].sum()} returned")
                else:
                    print(f"  Window {i}: Insufficient data")
            except Exception as e:
                print(f"  Window {i}: Error - {e}")
                continue
        
        if train_parts:
            train_combined = pd.concat(train_parts, ignore_index=True)
            test_features = prepare_test_features(data)
        else:
            print("Time windows failed, falling back to simple approach...")
            train_combined, test_features = create_simple_train_test_split(data)
    else:
        print("Using simple train/test split approach...")
        train_combined, test_features = create_simple_train_test_split(data)
    
    print(f"Training data shape: {train_combined.shape}")
    print(f"Users who returned/purchased: {train_combined['returned'].sum()}/{len(train_combined)}")
    
    if train_combined['returned'].sum() == 0:
        print("No users with purchases found. Creating synthetic targets based on purchase value...")
        # Create targets based on actual purchase values
        train_combined['target'] = np.log1p(train_combined.get('actual_purchase', train_combined.get('purchaseValue', 0)))
        train_combined['returned'] = (train_combined['target'] > 0).astype(int)
        print(f"Users with purchases: {train_combined['returned'].sum()}/{len(train_combined)}")
    
    print(f"Test data shape: {test_features.shape}")
    
    # Encode categorical features
    print("Encoding categorical features...")
    train_encoded, test_encoded, encoders = encode_categorical_features(
        train_combined, test_features
    )
    
    # Check if we have enough users who made purchases for training
    if train_encoded['returned'].sum() < 10:
        print("Warning: Very few users with purchases. Switching to direct regression...")
        return direct_regression_approach(train_encoded, test_encoded)
    
    # Validation split for R² evaluation
    from sklearn.model_selection import train_test_split
    train_for_model, val_for_eval = train_test_split(
        train_encoded, test_size=0.2, random_state=42, 
        stratify=train_encoded['returned'] if train_encoded['returned'].nunique() > 1 else None
    )
    
    # Train models
    print("Training models...")
    models_clf, models_reg, feature_cols = train_models(train_for_model)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_predictions = make_predictions(models_clf, models_reg, val_for_eval, feature_cols)
    val_r2 = evaluate_r2_score(val_for_eval['target'].values, val_predictions)
    print(f"Validation R² Score: {val_r2:.4f}")
    
    # Make predictions on test set
    print("Making final predictions...")
    predictions = make_predictions(models_clf, models_reg, test_encoded, feature_cols)
    
    # Create submission file
    submission = pd.DataFrame({
        'userId': test_encoded['userId'],
        'PredictedLogRevenue': predictions
    })
    
    print(f"Prediction summary:")
    print(submission['PredictedLogRevenue'].describe())
    print(f"Validation R² Score: {val_r2:.4f}")
    
    # Save results
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return submission

def create_simple_train_test_split(data):
    """
    Create a simple train/test split when time windows don't work
    """
    print("Creating simple train/test split...")
    
    # Sort by date and split 80/20
    data_sorted = data.sort_values(['userId', 'date'])
    
    # Get unique users and split them
    unique_users = data['userId'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    split_point = int(len(unique_users) * 0.8)
    train_users = unique_users[:split_point]
    test_users = unique_users[split_point:]
    
    train_data = data[data['userId'].isin(train_users)]
    test_data = data[data['userId'].isin(test_users)]
    
    # Create features for training data
    train_features = create_user_features(train_data, train_data['date'].min(), train_data['date'].max())
    
    # Create target based on actual purchases
    train_features['actual_purchase'] = train_data.groupby('userId')['purchaseValue'].sum().values
    train_features['target'] = np.log1p(train_features['actual_purchase'])
    train_features['returned'] = (train_features['actual_purchase'] > 0).astype(int)
    
    # Create features for test data
    test_features = create_user_features(test_data, test_data['date'].min(), test_data['date'].max())
    test_features['target'] = np.nan
    test_features['returned'] = np.nan
    
    return train_features, test_features

def direct_regression_approach(train_data, test_data):
    """
    Fallback direct regression when we don't have enough classification data
    """
    print("Using direct regression approach...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Prepare features
    feature_cols = [col for col in train_data.columns 
                   if col not in ['userId', 'target', 'returned', 'actual_purchase']]
    
    X = train_data[feature_cols].fillna(0)
    y = train_data['target'].fillna(0)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest (more robust than LightGBM for small datasets)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Validate
    val_pred = model.predict(X_val)
    val_r2 = evaluate_r2_score(y_val, val_pred)
    print(f"Validation R² Score: {val_r2:.4f}")
    
    # Predict on test
    X_test = test_data[feature_cols].fillna(0)
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'userId': test_data['userId'],
        'PredictedLogRevenue': predictions
    })
    
    print("Prediction summary:")
    print(submission['PredictedLogRevenue'].describe())
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return submission

if __name__ == "__main__":
    submission = main()