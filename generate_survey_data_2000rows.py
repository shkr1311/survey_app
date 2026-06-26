import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_survey_dataset_2000():
    n_samples = 2000
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
    ages[np.random.choice(n_samples, 20, replace=False)] = np.random.randint(-5, 0, 20)  # Negative
    ages[np.random.choice(n_samples, 12, replace=False)] = np.random.randint(150, 200, 12)  # Too high
    
    # 4. Gender (with missing values)
    genders = np.random.choice(['Male', 'Female', 'Other', 'Prefer not to say'], 
                               n_samples, p=[0.48, 0.48, 0.02, 0.02])
    genders = genders.astype(object)
    genders[np.random.choice(n_samples, 120, replace=False)] = np.nan
    
    # 5. Education
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD', 'Below High School'],
        n_samples, p=[0.35, 0.40, 0.18, 0.05, 0.02]
    )
    education = education.astype(object)
    education[np.random.choice(n_samples, 8, replace=False)] = 'Unknown Degree'
    education[np.random.choice(n_samples, 60, replace=False)] = np.nan
    
    # 6. Employment Status
    employment = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired'],
        n_samples, p=[0.55, 0.15, 0.10, 0.12, 0.08]
    )
    employment = employment.astype(object)
    employment[np.random.choice(n_samples, 40, replace=False)] = np.nan
    
    # 7. Annual Income (with outliers and missing)
    base_income = np.random.lognormal(10.5, 0.8, n_samples)
    income = np.clip(base_income, 15000, 500000)
    # Add outliers
    income[np.random.choice(n_samples, 16, replace=False)] = np.random.uniform(1000000, 3000000, 16)
    income[np.random.choice(n_samples, 12, replace=False)] = np.random.uniform(-50000, -100, 12)
    # Missing values
    income[np.random.choice(n_samples, 160, replace=False)] = np.nan
    income = income.astype(float)
    
    # 8. Region
    regions = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'],
        n_samples, p=[0.22, 0.20, 0.18, 0.25, 0.15]
    )
    regions = regions.astype(object)
    regions[np.random.choice(n_samples, 32, replace=False)] = np.nan
    
    # 9. City Type
    city_types = np.random.choice(
        ['Metro', 'Tier-1', 'Tier-2', 'Tier-3', 'Rural'],
        n_samples, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    city_types = city_types.astype(object)
    city_types[np.random.choice(n_samples, 40, replace=False)] = np.nan
    
    # 10. Customer Satisfaction (1-10 scale)
    satisfaction = np.random.normal(7, 1.5, n_samples)
    satisfaction = np.clip(satisfaction, 1, 10).round(1)
    # Invalid values
    satisfaction[np.random.choice(n_samples, 12, replace=False)] = np.random.uniform(11, 15, 12)
    satisfaction[np.random.choice(n_samples, 8, replace=False)] = 0
    satisfaction[np.random.choice(n_samples, 80, replace=False)] = np.nan
    
    # 11. Product Usage Frequency (times per month)
    usage_frequency = np.random.poisson(8, n_samples).astype(float)
    usage_frequency = np.clip(usage_frequency, 0, 30)
    # Outliers
    usage_frequency[np.random.choice(n_samples, 16, replace=False)] = np.random.randint(100, 500, 16)
    usage_frequency[np.random.choice(n_samples, 100, replace=False)] = np.nan
    
    # 12. Experience Years
    experience_years = np.random.exponential(3, n_samples)
    experience_years = np.clip(experience_years, 0, 20).round(1)
    # Invalid negative
    experience_years[np.random.choice(n_samples, 8, replace=False)] = np.random.uniform(-5, -0.1, 8)
    experience_years[np.random.choice(n_samples, 72, replace=False)] = np.nan
    
    # 13. Last Purchase Amount
    purchase_amount = np.random.gamma(2, 150, n_samples)
    purchase_amount = np.clip(purchase_amount, 10, 5000).round(2)
    # Outliers
    purchase_amount[np.random.choice(n_samples, 20, replace=False)] = np.random.uniform(10000, 30000, 20)
    purchase_amount[np.random.choice(n_samples, 120, replace=False)] = np.nan
    
    # 14. NPS Score (0-10)
    nps_score = np.random.choice(range(0, 11), n_samples, 
                                 p=[0.02, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20, 0.18, 0.13])
    nps_score = nps_score.astype(float)
    nps_score[np.random.choice(n_samples, 88, replace=False)] = np.nan
    
    # 15. Response Quality Score (0-100)
    response_quality = np.random.beta(8, 2, n_samples) * 100
    response_quality = response_quality.round(1)
    response_quality[np.random.choice(n_samples, 60, replace=False)] = np.nan
    
    # 16. Response Time (seconds)
    response_time = np.random.gamma(3, 60, n_samples).astype(int)
    response_time = np.clip(response_time, 30, 1800)
    # Suspicious responses
    response_time[np.random.choice(n_samples, 16, replace=False)] = np.random.randint(5, 20, 16)
    response_time[np.random.choice(n_samples, 16, replace=False)] = np.random.randint(3600, 7200, 16)
    
    # 17. Sampling Weight
    weights = np.ones(n_samples)
    
    for i in range(n_samples):
        if 18 <= ages[i] <= 25:
            weights[i] *= 1.5
        elif 55 <= ages[i] <= 85:
            weights[i] *= 1.3
        
        if genders[i] == 'Other' or genders[i] == 'Prefer not to say':
            weights[i] *= 2.0
        
        if regions[i] == 'Central':
            weights[i] *= 1.4
        elif regions[i] == 'North':
            weights[i] *= 0.9
        
        if city_types[i] == 'Rural':
            weights[i] *= 2.5
        elif city_types[i] == 'Metro':
            weights[i] *= 0.8
            
    weights = weights / weights.mean()
    
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

if __name__ == '__main__':
    df = generate_survey_dataset_2000()
    output_file = 'sample_survey_data_2000rows.csv'
    df.to_csv(output_file, index=False)
    print("="*70)
    print(f"✓ DATASET GENERATED SUCCESSFULLY! Saved to: {output_file}")
    print("="*70)
