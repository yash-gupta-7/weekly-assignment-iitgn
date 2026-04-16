import pandas as pd
import numpy as np

def audit_data(df):
    """
    Conducts a thorough data quality audit across all columns.
    Returns a dictionary of issues found.
    """
    issues = {}
    
    # 1. Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        issues['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    # 2. Audit 'age' column
    # Should be numeric and within reasonable range (0-120)
    age_numeric = pd.to_numeric(df['age'], errors='coerce')
    if age_numeric.isnull().any():
        issues['age_non_numeric'] = f"Found {age_numeric.isnull().sum()} non-numeric values in 'age'"
    
    outliers = age_numeric[(age_numeric < 0) | (age_numeric > 120)]
    if not outliers.empty:
        issues['age_outliers'] = f"Ages outside 0-120 found: {outliers.unique()}"
            
    # 3. Audit 'bmi' column
    # Look for missing, zeros, or extreme outliers
    if 'bmi' in df.columns:
        bmi_numeric = pd.to_numeric(df['bmi'], errors='coerce')
        null_bmi = bmi_numeric.isnull().sum()
        zero_bmi = (bmi_numeric == 0).sum()
        extreme_bmi = bmi_numeric[(bmi_numeric < 10) | (bmi_numeric > 60)].count()
        if null_bmi > 0 or zero_bmi > 0 or extreme_bmi > 0:
            issues['bmi_issues'] = {
                'missing_or_non_numeric': int(null_bmi),
                'zeros': int(zero_bmi),
                'extreme_values': int(extreme_bmi)
            }
            
    # 4. Audit 'gender' column
    # Look for inconsistent labels
    if 'gender' in df.columns:
        unique_genders = df['gender'].unique()
        if len(unique_genders) > 3: # Expect M, F, Unknown or similar
             issues['gender_inconsistency'] = list(unique_genders)
             
    # 5. Audit 'department' column
    if 'department' in df.columns:
        unique_dept = df['department'].unique()
        issues['department_variants'] = list(unique_dept)
        
    # 6. Audit 'admission_date'
    # Check for consistency in format
    # (Simplified check: count variants if possible)
    
    # 7. Audit 'readmitted_30d' (Target)
    if 'readmitted_30d' in df.columns:
        class_dist = df['readmitted_30d'].value_counts(normalize=True).to_dict()
        issues['target_distribution'] = class_dist

    return issues

def clean_data(df):
    """
    Applies a principled data cleaning strategy.
    """
    df_clean = df.copy()
    
    # 1. Fix Gender (Normalize to M, F, U)
    gender_map = {
        'M': 'M', 'm': 'M', 'Male': 'M', 'male': 'M',
        'F': 'F', 'f': 'F', 'Female': 'F', 'female': 'F',
        'Unknown': 'U', 'U': 'U', 'unknown': 'U'
    }
    df_clean['gender'] = df_clean['gender'].map(gender_map).fillna('U')
    
    # 2. Fix Department (Lower case)
    df_clean['department'] = df_clean['department'].str.lower().str.strip()
    
    # 3. Fix BMI (Impute missing with median, clip extreme outliers)
    df_clean['bmi'] = pd.to_numeric(df_clean['bmi'], errors='coerce')
    bmi_median = df_clean['bmi'].median()
    df_clean['bmi'] = df_clean['bmi'].fillna(bmi_median)
    df_clean['bmi'] = df_clean['bmi'].clip(lower=10, upper=60)
    
    # 4. Fix Age
    # Ensure numeric
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    age_median = df_clean['age'].median()
    df_clean['age'] = df_clean['age'].fillna(age_median)
    df_clean['age'] = df_clean['age'].clip(lower=0, upper=100)
    
    # 5. Handle missing in other numeric columns
    # First, try to convert columns that look numeric
    for col in df_clean.columns:
        if col not in ['gender', 'department', 'insurance_type', 'admission_date', 'patient_id']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # 6. Drop Patient ID if it's messy or irrelevant for modeling
    # But wait, XXXX might indicate a problem. Let's just strip it if needed.
    # For modeling, we don't need IDs.
    if 'patient_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['patient_id'])
        
    # 7. Convert Date to useful features
    if 'admission_date' in df_clean.columns:
        # Try multiple formats
        df_clean['admission_date'] = pd.to_datetime(df_clean['admission_date'], errors='coerce')
        # Fill missing dates if any
        df_clean['admission_date'] = df_clean['admission_date'].fillna(method='ffill')
        # Extract features
        df_clean['admission_month'] = df_clean['admission_date'].dt.month
        df_clean['admission_day_of_week'] = df_clean['admission_date'].dt.dayofweek
        df_clean = df_clean.drop(columns=['admission_date'])

    # 8. Encoding Categorical
    categorical_cols = ['gender', 'department', 'insurance_type']
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    return df_clean
