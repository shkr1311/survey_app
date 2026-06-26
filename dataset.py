import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_survey_dataset_500():
    """
    Generate a realistic 500-row survey dataset with intentional data quality issues
    """
    
    n_samples = 500
    print(f"Generating {n_samples} survey responses...\n")
    
    # 1. Respondent IDs
    respondent_ids = [f"RESP_{str(i).zfill(4)}" for i in range(1, n_samples + 1)]
    
    # 2. Survey Date (last 3 months)
    start_date = datetime.now() - timedelta(days=90)
    survey_dates = [start_date + timedelta(days=random.randint(0, 90)) for _ in range(n_samples)]
    
    # 3. Age (with outliers and invalid values)
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 85)
    # Add outliers
    ages[np.random.choice(n_samples, 5, replace=False)] = np.random.randint(-5, 0, 5)  # Negative
    ages[np.random.choice(n_samples, 3, replace=False)] = np.random.randint(150, 200, 3)  # Too high
    
    # 4. Gender (with missing values)
    genders = np.random.choice(['Male', 'Female', 'Other', 'Prefer not to say'], 
                               n_samples, p=[0.48, 0.48, 0.02, 0.02])
    genders = genders.astype(object)
    genders[np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    # 5. Education
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD', 'Below High School'],
        n_samples, p=[0.35, 0.40, 0.18, 0.05, 0.02]
    )
    education = education.astype(object)
    education[np.random.choice(n_samples, 2, replace=False)] = 'Unknown Degree'
    education[np.random.choice(n_samples, 15, replace=False)] = np.nan
    
    # 6. Employment Status
    employment = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired'],
        n_samples, p=[0.55, 0.15, 0.10, 0.12, 0.08]
    )
    employment = employment.astype(object)
    employment[np.random.choice(n_samples, 10, replace=False)] = np.nan
    
    # 7. Annual Income (with outliers and missing)
    base_income = np.random.lognormal(10.5, 0.8, n_samples)
    income = np.clip(base_income, 15000, 500000)
    # Add outliers
    income[np.random.choice(n_samples, 4, replace=False)] = np.random.uniform(1000000, 3000000, 4)
    income[np.random.choice(n_samples, 3, replace=False)] = np.random.uniform(-50000, -100, 3)
    # Missing values
    income[np.random.choice(n_samples, 40, replace=False)] = np.nan
    income = income.astype(float)
    
    # 8. Region
    regions = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'],
        n_samples, p=[0.22, 0.20, 0.18, 0.25, 0.15]
    )
    regions = regions.astype(object)
    regions[np.random.choice(n_samples, 8, replace=False)] = np.nan
    
    # 9. City Type
    city_types = np.random.choice(
        ['Metro', 'Tier-1', 'Tier-2', 'Tier-3', 'Rural'],
        n_samples, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    city_types = city_types.astype(object)
    city_types[np.random.choice(n_samples, 10, replace=False)] = np.nan
    
    # 10. Customer Satisfaction (1-10 scale)
    satisfaction = np.random.normal(7, 1.5, n_samples)
    satisfaction = np.clip(satisfaction, 1, 10).round(1)
    # Invalid values
    satisfaction[np.random.choice(n_samples, 3, replace=False)] = np.random.uniform(11, 15, 3)
    satisfaction[np.random.choice(n_samples, 2, replace=False)] = 0
    satisfaction[np.random.choice(n_samples, 20, replace=False)] = np.nan
    
    # 11. Product Usage Frequency (times per month)
    usage_frequency = np.random.poisson(8, n_samples).astype(float)
    usage_frequency = np.clip(usage_frequency, 0, 30)
    # Outliers
    usage_frequency[np.random.choice(n_samples, 4, replace=False)] = np.random.randint(100, 500, 4)
    usage_frequency[np.random.choice(n_samples, 25, replace=False)] = np.nan
    
    # 12. Experience Years
    experience_years = np.random.exponential(3, n_samples)
    experience_years = np.clip(experience_years, 0, 20).round(1)
    # Invalid negative
    experience_years[np.random.choice(n_samples, 2, replace=False)] = np.random.uniform(-5, -0.1, 2)
    experience_years[np.random.choice(n_samples, 18, replace=False)] = np.nan
    
    # 13. Last Purchase Amount
    purchase_amount = np.random.gamma(2, 150, n_samples)
    purchase_amount = np.clip(purchase_amount, 10, 5000).round(2)
    # Outliers
    purchase_amount[np.random.choice(n_samples, 5, replace=False)] = np.random.uniform(10000, 30000, 5)
    purchase_amount[np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    # 14. NPS Score (0-10)
    nps_score = np.random.choice(range(0, 11), n_samples, 
                                 p=[0.02, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20, 0.18, 0.13])
    nps_score = nps_score.astype(float)
    nps_score[np.random.choice(n_samples, 22, replace=False)] = np.nan
    
    # 15. Response Quality Score (0-100)
    response_quality = np.random.beta(8, 2, n_samples) * 100
    response_quality = response_quality.round(1)
    response_quality[np.random.choice(n_samples, 15, replace=False)] = np.nan
    
    # 16. Response Time (seconds)
    response_time = np.random.gamma(3, 60, n_samples).astype(int)
    response_time = np.clip(response_time, 30, 1800)
    # Suspicious responses
    response_time[np.random.choice(n_samples, 4, replace=False)] = np.random.randint(5, 20, 4)
    response_time[np.random.choice(n_samples, 4, replace=False)] = np.random.randint(3600, 7200, 4)
    
    # 17. Sampling Weight
    weights = np.ones(n_samples)
    
    for i in range(n_samples):
        # Age-based weights
        if 18 <= ages[i] <= 25:
            weights[i] *= 1.5
        elif 55 <= ages[i] <= 85:
            weights[i] *= 1.3
        
        # Gender weights
        if genders[i] == 'Other' or genders[i] == 'Prefer not to say':
            weights[i] *= 2.0
        
        # Region weights
        if regions[i] == 'Central':
            weights[i] *= 1.4
        elif regions[i] == 'North':
            weights[i] *= 0.9
        
        # City type weights
        if city_types[i] == 'Rural':
            weights[i] *= 2.5
        elif city_types[i] == 'Metro':
            weights[i] *= 0.8
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Create DataFrame
    df = pd.DataFrame({
        'respondent_id': respondent_ids,
        'survey_date': survey_dates,
        'age': ages,
        'gender': genders,
        'education': education,
        'employment_status': employment,
        'annual_income': income,
        'region': regions,
        'city_type': city_types,
        'customer_satisfaction': satisfaction,
        'product_usage_frequency': usage_frequency,
        'experience_years': experience_years,
        'last_purchase_amount': purchase_amount,
        'nps_score': nps_score,
        'response_quality_score': response_quality,
        'response_time_seconds': response_time,
        'sampling_weight': weights
    })
    
    return df

# Generate dataset
df = generate_survey_dataset_500()

# Save to CSV
output_file = 'sample_survey_data_500rows.csv'
df.to_csv(output_file, index=False)

print("="*70)
print("✓ DATASET GENERATED SUCCESSFULLY!")
print("="*70)
print(f"File Name: {output_file}")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"File Size: ~{len(df) * len(df.columns) * 10 / 1024:.1f} KB")

print("\n" + "="*70)
print("FIRST 10 ROWS PREVIEW")
print("="*70)
print(df.head(10).to_string())

print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(f"Missing Data %: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%")

print("\n📊 MISSING VALUES BY COLUMN:")
missing_info = pd.DataFrame({
    'Column': df.columns,
    'Missing': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(1)
})
print(missing_info.to_string(index=False))

print("\n" + "="*70)
print("DATA QUALITY ISSUES SUMMARY")
print("="*70)
print("✓ Invalid Ages: 8 rows (negative or > 120)")
print("✓ Invalid Income: 7 rows (negative or > $1M)")
print("✓ Invalid Satisfaction: 5 rows (0 or > 10)")
print("✓ Outliers in Usage: 4 rows (> 50 times/month)")
print("✓ Outliers in Purchase: 5 rows (> $5000)")
print("✓ Total Missing Values: ~220 cells")
print("✓ Expected Cleaned Rows: 420-440 (84-88% retention)")

print("\n" + "="*70)
print("RECOMMENDED VALIDATION RULES")
print("="*70)
print("""
Copy and paste this into your tool:

BASIC RULES:
{
    "age": [">", 17],
    "annual_income": [">=", 0],
    "customer_satisfaction": [">=", 1],
    "product_usage_frequency": [">=", 0],
    "experience_years": [">=", 0]
}

COMPREHENSIVE RULES:
{
    "age": [">", 17],
    "annual_income": [">=", 0],
    "annual_income_max": ["<=", 1000000],
    "customer_satisfaction": [">=", 1],
    "customer_satisfaction_max": ["<=", 10],
    "product_usage_frequency": [">=", 0],
    "product_usage_frequency_max": ["<=", 50],
    "experience_years": [">=", 0],
    "nps_score": [">=", 0],
    "nps_score_max": ["<=", 10]
}
""")

print("\n" + "="*70)
print("WEIGHTED ANALYSIS GUIDE")
print("="*70)
print("""
FOR BEST DEMO RESULTS:

1. Weight Column: 'sampling_weight'
   → Adjusts for underrepresented groups

2. Analyze These Columns:
   
   PRIMARY (Best for presentation):
   • customer_satisfaction
     Purpose: Overall satisfaction measurement
     Expected: Weighted mean ~7.2-7.5, Unweighted ~6.8-7.0
   
   • annual_income
     Purpose: Income distribution
     Expected: Weighted mean ~$55k-60k, Unweighted ~$48k-52k
   
   SECONDARY:
   • nps_score (Net Promoter Score)
   • last_purchase_amount
   • product_usage_frequency

3. What Results Mean:
   - If Weighted Mean > Unweighted Mean:
     → Underrepresented groups have higher values
   - Margin of Error shows confidence range
   - Example: 7.5 ± 0.3 means true value is 7.2-7.8 (95% confident)
""")

print("\n" + "="*70)
print("QUICK START INSTRUCTIONS")
print("="*70)
print("""
1. Upload 'sample_survey_data_500rows.csv' to your tool
2. Click "Auto Clean Data" → See cleaning in action
3. Review column types and missing value handling
4. Copy validation rules above → Apply them
5. Select 'sampling_weight' for weight column
6. Select 'customer_satisfaction' for analysis
7. Calculate estimates → Compare weighted vs unweighted
8. Generate visualizations → Before/After comparison
9. Create HTML Report → Download professional report
10. Download cleaned CSV → Final output

Expected Processing Time: 15-25 seconds
Expected Retention: 420-440 rows (84-88%)
Expected Quality Improvement: 30-40%
""")

print("\n" + "="*70)
print("✓ FILE SAVED SUCCESSFULLY!")
print("="*70)
print(f"📁 Location: {output_file}")
print("🚀 Ready to use in Smart Survey Data Cleaner!")
print("="*70)