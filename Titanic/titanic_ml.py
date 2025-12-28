#Titanic - Machine Learning from Disaster
#By: N4th4nGr33n
#Date: 12-27-25

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Set style for plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Load training and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Training data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)
    print("\nFirst few rows of training data:")
    print(train_df.head())
    
    return train_df, test_df


def explore_data(train_df):
    """Perform exploratory data analysis."""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print("\nTraining Data Info:")
    print(train_df.info())
    
    print("\nMissing Values:")
    print(train_df.isnull().sum())
    
    print("\nBasic Statistics:")
    print(train_df.describe())
    
    print("\nSurvival Distribution:")
    print(train_df['Survived'].value_counts())
    print("Survival Rate:", f"{train_df['Survived'].mean():.2%}")
    
    # Plot survival distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Survival by Sex
    train_df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 0], color=['blue', 'pink'])
    axes[0, 0].set_title('Survival Rate by Gender')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_xticklabels(['Female', 'Male'], rotation=0)
    
    # Survival by Pclass (Passenger Class)
    train_df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0, 1], color=['gold', 'silver', 'brown'])
    axes[0, 1].set_title('Survival Rate by Passenger Class')
    axes[0, 1].set_ylabel('Survival Rate')
    axes[0, 1].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
    
    # Survival by Age
    train_df.groupby(pd.cut(train_df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 80]))['Survived'].mean().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Survival Rate by Age Group')
    axes[1, 0].set_ylabel('Survival Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Survival by Fare
    train_df.boxplot(column='Fare', by='Survived', ax=axes[1, 1])
    axes[1, 1].set_title('Fare Distribution by Survival')
    axes[1, 1].set_xlabel('Survived')
    axes[1, 1].set_ylabel('Fare')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=100, bbox_inches='tight')
    print("\nPlots saved to eda_plots.png")
    plt.close()


def preprocess_data(df, is_test=False):
    """Preprocess the data (handle missing values, feature engineering, encode categoricals)."""
    df = df.copy()
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'].fillna('Unknown', inplace=True)
    
    
    Extract Title from Name (e.g. Mr, Mrs, Miss)
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs',
        'Lady': 'Royal',
        'Countess': 'Royal',
        'Sir': 'Royal',
        'Dona': 'Royal',
        'Jonkheer': 'Rare',
        'Capt': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Dr': 'Rare',
        'Rev': 'Rare'
    })
    
    # Family size features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Ticket prefix length (rough proxy for ticket type)
    df['TicketPrefix'] = df['Ticket'].str.replace('\.', '', regex=True)
    df['TicketPrefix'] = df['TicketPrefix'].str.replace('/', '', regex=True)
    df['TicketPrefix'] = df['TicketPrefix'].str.extract(r'(\D*)', expand=False).str.strip()
    df['TicketPrefix'] = df['TicketPrefix'].replace('', 'NONE')
    
    # Cabin deck (first letter of cabin)
    df['CabinDeck'] = df['Cabin'].str[0]
    df.loc[df['CabinDeck'] == 'U', 'CabinDeck'] = 'Unknown'
    
    # Age bands (to help tree-based models)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 45, 60, 80], labels=False, include_lowest=True)
    
    # Fare bands
    df['FareBand'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Ticket']
    # Keep Cabin only via CabinDeck feature
    if 'Cabin' in df.columns:
        drop_cols.append('Cabin')
    df = df.drop(drop_cols, axis=1)
    
    #Encode categorical variables
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # One-hot encode engineered categorical features
    cat_cols = ['Title', 'TicketPrefix', 'CabinDeck']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def train_models(X_train, X_val, y_train, y_val):
    #Train and evaluate multiple models.
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    results = {}
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)
    lr_accuracy = accuracy_score(y_val, y_pred_lr)
    
    print(f"Accuracy: {lr_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_lr))
    
    results['Logistic Regression'] = {'model': lr, 'scaler': scaler, 'accuracy': lr_accuracy}
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    rf_accuracy = accuracy_score(y_val, y_pred_rf)
    
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_rf))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    results['Random Forest'] = {'model': rf, 'scaler': None, 'accuracy': rf_accuracy}
    
    return results


def generate_submission(test_df, best_model, scaler=None, model_name='Random Forest', feature_cols=None):
    #Generate submission file.
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    # Preprocess test data
    test_processed = preprocess_data(test_df, is_test=True)
    
    # Extract PassengerId
    test_ids = test_df['PassengerId']
    
    # Align test features with training features to avoid feature name mismatches
    if feature_cols is not None:
        # Reindex ensures it has exactly the same columns (order and names)
        X_test = test_processed.reindex(columns=feature_cols, fill_value=0)
    else:
        X_test = test_processed
    
    # Make predictions
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': y_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission file created: submission.csv")
    print(f"Model used: {model_name}")
    print(f"\nFirst few rows:")
    print(submission.head(10))
    
    return submission


def main():
    print("="*50)
    print("TITANIC - MACHINE LEARNING FROM DISASTER")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    # Explore data
    explore_data(train_df)
    
    # Preprocess training data
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    train_processed = preprocess_data(train_df)
    print("Training data preprocessed")
    
    # Prepare features and target
    X = train_processed.drop('Survived', axis=1)
    y = train_processed['Survived']
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # Train models
    results = train_models(X_train, X_val, y_train, y_val)
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_scaler = results[best_model_name]['scaler']
    
    print("\n" + "="*50)
    print(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    print("="*50)
    
    # Generate submission
    # Pass the training feature columns so the test set can be aligned properly
    generate_submission(test_df, best_model, best_scaler, best_model_name, feature_cols=X.columns)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)


if __name__ == '__main__':
    main()
