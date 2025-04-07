import numpy as np
import pandas as pd
import random
import calendar
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
num_months = 36
unique_customers = 27778  # Fixed set of unique customers
total_records = unique_customers * num_months  # 1,000,008 records
churn_threshold = 0.6  # Churn risk score threshold for churn event

# ---------------------------
# Helper Function: Last Day of Month
# ---------------------------
def last_day_of_month(dt: pd.Timestamp) -> datetime:
    year, month = dt.year, dt.month
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, last_day)

# ---------------------------
# Calculate Churn Risk Score
# ---------------------------
def calculate_churn_risk(df: pd.DataFrame) -> pd.Series:
    # Usage Patterns
    # Lower call minutes, data usage, SMS, or internet sessions indicate lower engagement.
    risk_call = (600 - df["CallMinutes"]) / 500  # Expected range: 100 to 600
    risk_data = (100 - df["DataUsageGB"]) / 99     # Generated range: 1 to 100
    risk_sms = (200 - df["SMSCount"]) / 200         # Generated range: 0 to 200
    risk_internet = (100 - df["InternetSessions"]) / 100  # Generated range: 0 to 100

    # Billing Information
    # Higher bill amounts and late fees increase risk.
    risk_bill = (df["TotalBillAmount"] - 20) / 130  # Generated range: 20 to 150
    payment_mapping = {"Paid": 0, "Partial": 0.5, "Unpaid": 1}
    risk_payment = df["PaymentStatus"].map(payment_mapping)
    risk_late_fee = df["LateFee"] / 20              # Generated range: 0 to ~20
    # Receiving discounts should lower risk.
    risk_discount = 1 - (df["DiscountsApplied"] / 20)  # Generated range: 0 to 20

    # Customer Support Interactions
    risk_support = df["SupportInteractionCount"] / 5  # Generated range: 0 to 5
    # Map issue types to a risk factor. If no interaction, risk is 0.
    issue_mapping = {"": 0, "Billing": 0.7, "Technical": 0.8, "Service Quality": 0.6}
    risk_issue = df["PrimaryIssueType"].map(issue_mapping).fillna(0)
    # Longer resolution times are riskier; resolution time expected between 10 and 120 minutes.
    risk_resolution = (df["ResolutionTimeMinutes"] - 10) / 110
    risk_resolution = risk_resolution.clip(lower=0)  # Ensure negative values don't occur.
    # Lower satisfaction increases risk.
    risk_satisfaction = (5 - df["SatisfactionScore"]) / 4
    # If an interaction was escalated, add risk.
    risk_escalation = df["EscalationFlag"].astype(int)  # True->1, False->0

    # Define weights for each factor.
    w_call = 0.1
    w_data = 0.1
    w_sms = 0.05
    w_internet = 0.05
    w_bill = 0.1
    w_payment = 0.1
    w_late_fee = 0.05
    w_discount = 0.05
    w_support = 0.1
    w_issue = 0.05
    w_resolution = 0.05
    w_satisfaction = 0.15
    w_escalation = 0.1

    # Combine all risk factors into a final risk score.
    risk_score = (
        w_call * risk_call +
        w_data * risk_data +
        w_sms * risk_sms +
        w_internet * risk_internet +
        w_bill * risk_bill +
        w_payment * risk_payment +
        w_late_fee * risk_late_fee +
        w_discount * risk_discount +
        w_support * risk_support +
        w_issue * risk_issue +
        w_resolution * risk_resolution +
        w_satisfaction * risk_satisfaction +
        w_escalation * risk_escalation
    )

    # Inject random noise to lower the predictive accuracy (e.g., to around 75%)
    # Adjust the scale parameter as needed to achieve the desired accuracy.
    noise = np.random.normal(loc=0, scale=0.25, size=risk_score.shape)
    noisy_risk_score = risk_score + noise

    # Ensure the risk score is within [0, 1] and round to two decimals.
    return noisy_risk_score.clip(0, 1).round(2)

def filter_entries_after_churn(
    data: pd.DataFrame,
    date_column: str = "BillingCycleStart",
    churn_flag: str = "HasChurned",
    customer_id_column: str = "CustomerID"
) -> pd.DataFrame:
    """
    For each customer, remove records that occur after the first occurrence of a churn event.
    The record where HasChurned becomes True is retained, but all subsequent records for that customer are filtered out.

    Parameters:
        data (pd.DataFrame): Input DataFrame with at least customer ID, a date column, and a churn flag.
        date_column (str): The column name to use for ordering records (e.g., "BillingCycleStart").
        churn_flag (str): The column name indicating if the customer has churned (boolean).
        customer_id_column (str): The column name for the customer identifier.

    Returns:
        pd.DataFrame: A filtered DataFrame with entries after the churn event removed.
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Ensure the date column is in datetime format and sort the DataFrame
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=[customer_id_column, date_column])
    
    def filter_customer_group(group: pd.DataFrame) -> pd.DataFrame:
        if group[churn_flag].any():
            # Find the first occurrence of churn
            first_churn_date = group.loc[group[churn_flag], date_column].iloc[0]
            # Keep records up to and including the churn event
            return group[group[date_column] <= first_churn_date]
        else:
            return group

    # Apply the filtering function for each customer
    filtered_df = df.groupby(customer_id_column, group_keys=False).apply(filter_customer_group)
    return filtered_df


# ---------------------------
# Generate List of Billing Cycle Start Dates for 36 Months
# ---------------------------
billing_cycle_starts = pd.date_range(start="2020-01-01", periods=num_months, freq="MS")

# Pre-generate unique customer IDs (the same set for each month to simulate time series)
customer_ids = [f"CUST{i:07d}" for i in range(1, unique_customers + 1)]

# List to store DataFrame for each month
monthly_dfs = []

# ---------------------------
# Generate Data Month by Month
# ---------------------------
for month_start in billing_cycle_starts:
    billing_start = month_start
    billing_end = pd.Timestamp(last_day_of_month(month_start))
    payment_due_date = billing_end + pd.Timedelta(days=5)
    
    # ---------------------------
    # Usage Patterns
    # ---------------------------
    call_minutes = np.random.randint(100, 600, size=unique_customers)
    data_usage = np.round(np.random.uniform(1, 100, size=unique_customers), 2)
    sms_count = np.random.randint(0, 200, size=unique_customers)
    internet_sessions = np.random.randint(0, 100, size=unique_customers)
    
    # ---------------------------
    # Billing Information
    # ---------------------------
    total_bill_amount = np.round(np.random.uniform(20, 150, size=unique_customers), 2)
    payment_status_options = ["Paid", "Unpaid", "Partial"]
    payment_status_probs = [0.8, 0.1, 0.1]
    payment_status = np.random.choice(payment_status_options, size=unique_customers, p=payment_status_probs)
    
    # Late fee: based on payment status.
    late_fee = []
    for status in payment_status:
        if status == "Paid":
            fee = 0.0
        elif status == "Unpaid":
            fee = round(random.uniform(5, 20), 2)
        else:  # Partial
            fee = round(random.uniform(1, 10), 2)
        late_fee.append(fee)
    late_fee = np.array(late_fee)
    
    discounts_applied = np.round(np.random.uniform(0, 20, size=unique_customers), 2)
    
    # ---------------------------
    # Customer Support Interactions
    # ---------------------------
    support_interaction_count = np.random.randint(0, 6, size=unique_customers)
    
    last_interaction_date = []
    primary_issue_type = []
    resolution_time = []
    satisfaction_score = []
    escalation_flag = []
    support_channel = []
    
    issue_types = ["Billing", "Technical", "Service Quality"]
    support_channels = ["Phone", "Chat", "Email"]
    
    for count in support_interaction_count:
        if count > 0:
            # Random interaction date between billing_start and billing_end
            delta_days = (billing_end - billing_start).days
            random_day = random.randint(0, delta_days)
            interaction_date = billing_start + pd.Timedelta(days=random_day)
            last_interaction_date.append(interaction_date)
            primary_issue_type.append(random.choice(issue_types))
            resolution_time.append(random.randint(10, 120))
            satisfaction_score.append(random.randint(1, 5))
            escalation_flag.append(random.random() < 0.1)  # 10% chance of escalation
            support_channel.append(random.choice(support_channels))
        else:
            last_interaction_date.append(pd.NaT)
            primary_issue_type.append("")
            resolution_time.append(0)
            satisfaction_score.append(5)  # Default high satisfaction if no interaction
            escalation_flag.append(False)
            support_channel.append("")
    
    # ---------------------------
    # Create DataFrame for the Month
    # ---------------------------
    df_month = pd.DataFrame({
        "CustomerID": customer_ids,  # Same set for every month
        "BillingCycleStart": billing_start,
        "BillingCycleEnd": billing_end,
        "CallMinutes": call_minutes,
        "DataUsageGB": data_usage,
        "SMSCount": sms_count,
        "InternetSessions": internet_sessions,
        "TotalBillAmount": total_bill_amount,
        "PaymentStatus": payment_status,
        "PaymentDueDate": payment_due_date,
        "LateFee": late_fee,
        "DiscountsApplied": discounts_applied,
        "SupportInteractionCount": support_interaction_count,
        "LastInteractionDate": last_interaction_date,
        "PrimaryIssueType": primary_issue_type,
        "ResolutionTimeMinutes": resolution_time,
        "SatisfactionScore": satisfaction_score,
        "EscalationFlag": escalation_flag,
        "SupportChannel": support_channel
    })
    
    monthly_dfs.append(df_month)

# Concatenate all monthly DataFrames into one final DataFrame
df = pd.concat(monthly_dfs, ignore_index=True)
print(f"Total records generated: {len(df)}")

# Apply the churn risk function to compute the score for each record.
df["HasChurned"] = calculate_churn_risk(df) > churn_threshold

# Filter out records after the first churn event for each customer
df = filter_entries_after_churn(df)
print(f"Total records after filtering: {len(df)}")

churned_customers = df[df["HasChurned"]]
print(f"Number of churned customers: {len(churned_customers)}")
print(churned_customers)

# ---------------------------
# Save the Synthetic Data
# ---------------------------
data_path = "data/synthetic_customer_data_evenly_distributed.csv"
df.to_csv(data_path, index=False)
print(f"Synthetic data generated and saved to '{data_path}'")