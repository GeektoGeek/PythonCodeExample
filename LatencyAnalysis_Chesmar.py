# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:33:25 2020

@author: dsarkar
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Read the data files
sql_df = pd.read_csv('C:/Users/17024/Documents/CreativeCircle/Division_Analysis/OneDrive_2025-10-17/Chesmar/Sql_Processed.csv',encoding= 'iso-8859-1')
mql_df = pd.read_csv('C:/Users/17024/Documents/CreativeCircle/Division_Analysis/OneDrive_2025-10-17/Chesmar/mql.csv',encoding= 'iso-8859-1')


# Clean column names
mql_df.columns = mql_df.columns.str.strip()
sql_df.columns = sql_df.columns.str.strip()

print("="*80)
print("INITIAL DATA LOAD")
print("="*80)
print(f"MQL Records: {len(mql_df)}")
print(f"SQL Records: {len(sql_df)}")

# Standardize Contact ID
mql_df['Contact ID'] = mql_df['Contact ID'].astype(str).str.strip()
sql_df['Contact ID'] = sql_df['Contact ID'].astype(str).str.strip()

# Parse dates FIRST before merging
mql_df['MQL_Date'] = pd.to_datetime(
    mql_df['Date entered "Marketing Qualified Lead (Lifecycle Stage Pipeline)"'],
    format='%m/%d/%Y %H:%M',
    errors='coerce'
)
sql_df['SQL_Date'] = pd.to_datetime(
    sql_df['Date entered "Sales Qualified Lead (Lifecycle Stage Pipeline)"'],
    format='%m/%d/%Y %H:%M',
    errors='coerce'
)

# Define columns using the 'Chesmar' prefix for accurate lookup
community_columns_prefix = 'Chesmar'
community_columns = [
    ' - Austin Communities of Interest',
    ' - DFW West Communities of Interest',
    ' - DFW East Communities of Interest',
    ' - Houston South Communities of Interest',
    ' - Houston North Communities of Interest',
    ' - Central Texas Communities of Interest',
    ' - Houston West Communities of Interest',
    ' - San Antonio Communities of Interest'
]
full_community_columns = [f'{community_columns_prefix}{col}' for col in community_columns]

def extract_all_communities(row, suffix=''):
    """Extract all communities from a row"""
    communities = set()
    for col in full_community_columns:
        col_name = col + suffix if suffix else col
        if col_name in row.index:
            value = row[col_name]
            if pd.notna(value) and value != '(No value)' and str(value).strip() != '':
                for community in str(value).split(';'):
                    community = community.strip()
                    if community:
                        communities.add(community)
    return communities

def get_primary_region_from_communities(row, suffix=''):
    """Infer primary region from communities"""
    regions = []
    
    # Map communities to regions - checks for the correct full column name
    
    # Austin
    austin_col = community_columns_prefix + community_columns[0] + suffix
    if austin_col in row.index:
        austin_val = row.get(austin_col, '')
        if pd.notna(austin_val) and austin_val != '(No value)' and str(austin_val).strip() != '':
            regions.append('Austin')
    
    # DFW
    dfw_west_col = community_columns_prefix + community_columns[1] + suffix
    dfw_east_col = community_columns_prefix + community_columns[2] + suffix
    
    dfw_west_val = row.get(dfw_west_col, '') if dfw_west_col in row.index else ''
    dfw_east_val = row.get(dfw_east_col, '') if dfw_east_col in row.index else ''

    if (pd.notna(dfw_west_val) and dfw_west_val != '(No value)' and str(dfw_west_val).strip() != '') or \
       (pd.notna(dfw_east_val) and dfw_east_val != '(No value)' and str(dfw_east_val).strip() != ''):
        regions.append('DFW')
    
    # Houston
    hs_col = community_columns_prefix + community_columns[3] + suffix
    hn_col = community_columns_prefix + community_columns[4] + suffix
    hw_col = community_columns_prefix + community_columns[6] + suffix
    
    hs_val = row.get(hs_col, '') if hs_col in row.index else ''
    hn_val = row.get(hn_col, '') if hn_col in row.index else ''
    hw_val = row.get(hw_col, '') if hw_col in row.index else ''

    if (pd.notna(hs_val) and hs_val != '(No value)' and str(hs_val).strip() != '') or \
       (pd.notna(hn_val) and hn_val != '(No value)' and str(hn_val).strip() != '') or \
       (pd.notna(hw_val) and hw_val != '(No value)' and str(hw_val).strip() != ''):
        regions.append('Houston')
    
    # Central Texas
    ct_col = community_columns_prefix + community_columns[5] + suffix
    if ct_col in row.index:
        ct_val = row.get(ct_col, '')
        if pd.notna(ct_val) and ct_val != '(No value)' and str(ct_val).strip() != '':
            regions.append('Central Texas')
    
    # San Antonio
    sa_col = community_columns_prefix + community_columns[7] + suffix
    if sa_col in row.index:
        sa_val = row.get(sa_col, '')
        if pd.notna(sa_val) and sa_val != '(No value)' and str(sa_val).strip() != '':
            regions.append('San Antonio')
    
    return regions

# Add community extraction to both dataframes
mql_df['MQL_Communities'] = mql_df.apply(lambda x: extract_all_communities(x), axis=1)
sql_df['SQL_Communities'] = sql_df.apply(lambda x: extract_all_communities(x), axis=1)

mql_df['MQL_Regions'] = mql_df.apply(lambda x: get_primary_region_from_communities(x), axis=1)
sql_df['SQL_Regions'] = sql_df.apply(lambda x: get_primary_region_from_communities(x), axis=1)

# Merge on Contact ID first
merged_df = pd.merge(
    sql_df,
    mql_df,
    on='Contact ID',
    how='inner',
    suffixes=('_SQL', '_MQL')
)

print(f"\nInitial Merge (Contact ID only): {len(merged_df)} records")

# JOURNEY VALIDATION
print("\n" + "="*80)
print("JOURNEY VALIDATION")
print("="*80)

# 1. Temporal Validation: MQL must come before SQL
merged_df['Temporal_Valid'] = merged_df['MQL_Date'] < merged_df['SQL_Date']
temporal_invalid = (~merged_df['Temporal_Valid']).sum()
print(f"✓ Temporal Check: {temporal_invalid} records with MQL >= SQL (will be filtered)")

# 2. Calculate latency
merged_df['Latency_Days'] = (merged_df['SQL_Date'] - merged_df['MQL_Date']).dt.total_seconds() / (24 * 3600)
merged_df['Latency_Hours'] = (merged_df['SQL_Date'] - merged_df['MQL_Date']).dt.total_seconds() / 3600

# 3. Regional Consistency Check - FIXED VERSION
def check_regional_consistency(row):
    """Check if MQL and SQL regions are consistent"""
    mql_regions = row['MQL_Regions']
    sql_regions = row['SQL_Regions']
    
    # FIXED: Check for the column with the correct suffix AND correct prefix 'Chesmar'
    base_col = 'Chesmar Primary Regions of Interest'
    
    if base_col + '_MQL' in row.index:
        primary_region = row[base_col + '_MQL']
    elif base_col + '_SQL' in row.index:
        primary_region = row[base_col + '_SQL']
    elif base_col in row.index:
        primary_region = row[base_col]
    else:
        primary_region = None
    
    # If no communities in SQL, it's still valid (lead may not have specified yet)
    if len(sql_regions) == 0:
        return 'SQL_No_Region'
    
    # If no communities in MQL but has primary region
    if len(mql_regions) == 0 and pd.notna(primary_region) and str(primary_region).strip() != '':
        return 'MQL_Primary_Only'
    
    # Check if there's any overlap
    if len(set(mql_regions) & set(sql_regions)) > 0:
        return 'Consistent'
    
    # Check if primary region matches SQL regions
    if pd.notna(primary_region) and str(primary_region).strip() != '':
        for region in sql_regions:
            if region in str(primary_region) or str(primary_region) in region:
                return 'Primary_Matches'
    
    return 'Inconsistent'

merged_df['Regional_Consistency'] = merged_df.apply(check_regional_consistency, axis=1)

print("\nRegional Consistency Distribution:")
print(merged_df['Regional_Consistency'].value_counts())

# 4. Community Overlap Check
def calculate_community_overlap(row):
    """Calculate Jaccard similarity between MQL and SQL communities"""
    mql_comm = row['MQL_Communities']
    sql_comm = row['SQL_Communities']
    
    if len(mql_comm) == 0 and len(sql_comm) == 0:
        return 0.0  # No communities at all
    
    if len(sql_comm) == 0:
        return 0.5  # SQL has no communities yet, partial match
    
    if len(mql_comm) == 0:
        return 0.5  # MQL had no communities, partial match
    
    intersection = len(mql_comm & sql_comm)
    union = len(mql_comm | sql_comm)
    
    return intersection / union if union > 0 else 0.0

merged_df['Community_Overlap'] = merged_df.apply(calculate_community_overlap, axis=1)

print(f"\nAverage Community Overlap (Jaccard): {merged_df['Community_Overlap'].mean():.2%}")
print(f"Records with >0% overlap: {(merged_df['Community_Overlap'] > 0).sum()}")
print(f"Records with 100% overlap: {(merged_df['Community_Overlap'] == 1.0).sum()}")

# 5. Journey Quality Score
def calculate_journey_quality(row):
    """Calculate overall journey quality score (0-100)"""
    score = 0
    
    # Temporal validity (40 points)
    if row['Temporal_Valid']:
        score += 40
    
    # Regional consistency (30 points)
    if row['Regional_Consistency'] in ['Consistent', 'Primary_Matches']:
        score += 30
    elif row['Regional_Consistency'] in ['SQL_No_Region', 'MQL_Primary_Only']:
        score += 15
    
    # Community overlap (30 points)
    score += row['Community_Overlap'] * 30
    
    return score

merged_df['Journey_Quality_Score'] = merged_df.apply(calculate_journey_quality, axis=1)

# Quality categories
merged_df['Journey_Quality_Category'] = pd.cut(
    merged_df['Journey_Quality_Score'],
    bins=[0, 50, 70, 90, 100],
    labels=['Poor', 'Fair', 'Good', 'Excellent'],
    right=True
)

print("\nJourney Quality Distribution:")
print(merged_df['Journey_Quality_Category'].value_counts().sort_index())

# Filter for valid journeys
valid_journeys = merged_df[
    (merged_df['Temporal_Valid']) &
    (merged_df['Latency_Days'] >= 0) &
    (merged_df['Journey_Quality_Score'] >= 40)  # Minimum quality threshold
].copy()

print(f"\n✓ Valid Journeys After Validation: {len(valid_journeys)} / {len(merged_df)}")

# Aggregate communities
def aggregate_all_communities(row):
    """Aggregate all unique communities from MQL and SQL"""
    all_communities = row['MQL_Communities'] | row['SQL_Communities']
    return '; '.join(sorted(all_communities)) if all_communities else 'None'

valid_journeys['All_Communities'] = valid_journeys.apply(aggregate_all_communities, axis=1)

# Determine which Primary Region column to use - **FIXED** to include 'Chesmar'
base_col = 'Chesmar Primary Regions of Interest'
if base_col + '_MQL' in valid_journeys.columns:
    primary_region_col = base_col + '_MQL'
elif base_col + '_SQL' in valid_journeys.columns:
    primary_region_col = base_col + '_SQL'
else:
    # This is the corrected fallback for the full column name
    primary_region_col = base_col

# Create final unified dataset
unified_df = pd.DataFrame({
    'Contact_ID': valid_journeys['Contact ID'],
    'Primary_Region': valid_journeys[primary_region_col].fillna('Not Specified'),
    'MQL_Date': valid_journeys['MQL_Date'].dt.strftime('%m/%d/%Y %H:%M'),
    'SQL_Date': valid_journeys['SQL_Date'].dt.strftime('%m/%d/%Y %H:%M'),
    'Latency_Days': valid_journeys['Latency_Days'].round(2),
    'Latency_Hours': valid_journeys['Latency_Hours'].round(2),
    'MQL_Communities': valid_journeys['MQL_Communities'].apply(lambda x: '; '.join(sorted(x)) if x else 'None'),
    'SQL_Communities': valid_journeys['SQL_Communities'].apply(lambda x: '; '.join(sorted(x)) if x else 'None'),
    'All_Communities': valid_journeys['All_Communities'],
    'MQL_Traffic_Source': valid_journeys['Original Traffic Source Drill-Down 1_MQL'].fillna('Unknown'),
    'SQL_Traffic_Source': valid_journeys['Original Traffic Source Drill-Down 1_SQL'].fillna('Unknown'),
    'Regional_Consistency': valid_journeys['Regional_Consistency'],
    'Community_Overlap_Pct': (valid_journeys['Community_Overlap'] * 100).round(1),
    'Journey_Quality_Score': valid_journeys['Journey_Quality_Score'].round(1),
    'Journey_Quality_Category': valid_journeys['Journey_Quality_Category']
})

# SUMMARY STATISTICS
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total Validated Conversions: {len(unified_df)}")
print(f"Average Latency: {unified_df['Latency_Days'].mean():.2f} days")
print(f"Median Latency: {unified_df['Latency_Days'].median():.2f} days")
print(f"25th Percentile: {unified_df['Latency_Days'].quantile(0.25):.2f} days")
print(f"75th Percentile: {unified_df['Latency_Days'].quantile(0.75):.2f} days")
print(f"Min Latency: {unified_df['Latency_Days'].min():.2f} days")
print(f"Max Latency: {unified_df['Latency_Days'].max():.2f} days")
print(f"Average Journey Quality Score: {unified_df['Journey_Quality_Score'].mean():.1f}/100")

# Regional Performance
print("\n" + "="*80)
print("REGIONAL PERFORMANCE (with Journey Validation)")
print("="*80)
regional_stats = unified_df.groupby('Primary_Region').agg({
    'Contact_ID': 'count',
    'Latency_Days': ['mean', 'median', 'min', 'max'],
    'Journey_Quality_Score': 'mean',
    'Community_Overlap_Pct': 'mean'
}).round(2)
regional_stats.columns = ['Count', 'Avg_Latency', 'Median_Latency', 'Min_Latency',
                          'Max_Latency', 'Avg_Quality_Score', 'Avg_Community_Overlap']
regional_stats = regional_stats.sort_values('Count', ascending=False)
print(regional_stats)

# Journey Quality by Region
print("\n" + "="*80)
print("JOURNEY QUALITY BY REGION")
print("="*80)
quality_by_region = pd.crosstab(
    unified_df['Primary_Region'],
    unified_df['Journey_Quality_Category']
)
print(quality_by_region)

# Latency by Journey Quality
print("\n" + "="*80)
print("LATENCY BY JOURNEY QUALITY")
print("="*80)
latency_quality = unified_df.groupby('Journey_Quality_Category')['Latency_Days'].agg(['count', 'mean', 'median']).round(2)
# Reindex to ensure order
category_order = ['Poor', 'Fair', 'Good', 'Excellent']
latency_quality = latency_quality.reindex(category_order)
print(latency_quality)

# Traffic Source Analysis
print("\n" + "="*80)
print("TOP 10 TRAFFIC SOURCES (MQL to SQL)")
print("="*80)
source_stats = unified_df.groupby('MQL_Traffic_Source').agg({
    'Contact_ID': 'count',
    'Latency_Days': 'mean',
    'Journey_Quality_Score': 'mean'
}).round(2)
source_stats.columns = ['Conversions', 'Avg_Latency_Days', 'Avg_Quality_Score']
source_stats = source_stats.sort_values('Conversions', ascending=False).head(10)
print(source_stats)

# Community Consistency Analysis
print("\n" + "="*80)
print("COMMUNITY JOURNEY CONSISTENCY")
print("="*80)
print(f"Leads with identical communities (MQL→SQL): {(unified_df['Community_Overlap_Pct'] == 100).sum()}")
print(f"Leads with partial overlap (>0% and <100%): {((unified_df['Community_Overlap_Pct'] > 0) & (unified_df['Community_Overlap_Pct'] < 100)).sum()}")
print(f"Leads with no community overlap: {(unified_df['Community_Overlap_Pct'] == 0).sum()}")

# Save outputs (using accessible file names)
main_output = 'mql_to_sql_unified_validated.csv'
unified_df.to_csv(main_output, index=False)
print(f"\n✓ Main unified dataset saved to: {main_output}")

# Save validation report - **FIXED** to use the correct column name
base_col = 'Chesmar Primary Regions of Interest'
if base_col + '_MQL' in merged_df.columns:
    validation_primary_col = base_col + '_MQL'
elif base_col + '_SQL' in merged_df.columns:
    validation_primary_col = base_col + '_SQL'
else:
    validation_primary_col = base_col # Corrected fallback

validation_report = merged_df[['Contact ID', validation_primary_col,
                               'Temporal_Valid', 'Regional_Consistency',
                               'Community_Overlap', 'Journey_Quality_Score',
                               'Journey_Quality_Category', 'Latency_Days']].copy()

# Rename for clarity
validation_report.rename(columns={validation_primary_col: 'Primary_Regions_of_Interest'}, inplace=True)

validation_output = 'journey_validation_report.csv'
validation_report.to_csv(validation_output, index=False)
print(f"✓ Journey validation report saved to: {validation_output}")

# Display sample of validated journeys
print("\n" + "="*80)
print("SAMPLE VALIDATED JOURNEYS (First 15 Records)")
print("="*80)
display_cols = ['Contact_ID', 'Primary_Region', 'Latency_Days',
                'Community_Overlap_Pct', 'Journey_Quality_Score', 'Journey_Quality_Category']
print(unified_df[display_cols].head(15).to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE WITH JOURNEY VALIDATION!")
print("="*80)
print("\nKey Outputs:")
print(f"1. {main_output} - Validated customer journeys")
print(f"2. {validation_output} - Detailed validation metrics for all records")

Out5 = validation_report.to_csv(r'C:\Users\17024\Documents\CreativeCircle\Division_Analysis\OneDrive_2025-10-17\Chesmar\validation_report.csv', index = None, header=True)

Out5 = merged_df.to_csv(r'C:\Users\17024\Documents\CreativeCircle\Division_Analysis\OneDrive_2025-10-17\Chesmar\merged_df.csv', index = None, header=True)






