import pandas as pd
import re
import numpy as np

from datetime import datetime

CURRENT_YEAR = datetime.now().year

def extract_year(value):
    """
        - If the value is missing or invalid, assigns -1.
    """
    if pd.isna(value) or value in ["unknown", "N/A", "missing"]:
        return -1
    value = str(value).strip().lower()

    match = re.search(r"(\d{4})", value)
    if match:
        year = int(match.group(1))
        if year <= CURRENT_YEAR:
            return year
        else:
            return -1
    return -1

def standardize_culture(value):
    if pd.isna(value) or value in ["N/A", "missing"]:
        return "unknown"
    value = str(value).strip().lower()
    value = value.replace("probably", "").strip()
    value = value.replace("(american market)", "american market")
    value = value.replace(" or ", "/")
    value = value.replace(",", " -")
    value = re.sub(r"\s+", " ", value)
    return value

def standardize(value):
    if pd.isna(value) or value.strip() in ["|", "", " | ", "missing", " "]:
        return "unknown"

    value = str(value).strip().lower()
    value = re.sub(r"\s*\|\s*", ", ", value)
    value = re.sub(r"\s*\(\s*", " (", value)
    value = re.sub(r"\s*\)\s*", ")", value)
    value = re.sub(r"born\s+[a-zA-Z\s]+", "", value).strip()

    replace_dict = {
        "american|american": "american",
        "american | american": "american",
        "american|": "american",
        "|american": "american",
        "| american| american": "american",
        "| american": "american",
        "british|": "british",
        "|british": "british",
        "american,": "american",
        "american american": "american",
        "american , american": "american",
        "american ,": "american",
        ", american": "american",
        "Maker|Maker": "Maker",
        "Manufacturer|Maker": "Manufacturer",
        "Designer|Manufacturer|Maker": "Manufacturer",
    }

    for wrong, correct in replace_dict.items():
        value = value.replace(wrong, correct)

    value = re.sub(r"\(.*?\)", "", value).strip()
    value = value.strip(", ")

    return value

def clean_object_number(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9.]", "", value)
    value = value.lstrip(".")
    value = value.rstrip(".")
    value = re.sub(r"\.{2,}", ".", value)
    return value

def detect_outliers_iqr(df, column):
    """
    Since I changed some of the values per column I decided to ignore invalid (-1) values.
    """
    valid_data = df[df[column] != -1]
    Q1 = valid_data[column].quantile(0.25)
    Q3 = valid_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = valid_data[(valid_data[column] < lower_bound) | (valid_data[column] > upper_bound)]
    return outliers


def detect_outliers_zscore(df, column, threshold=3):
    """
    Since I changed some of the values per column I decided to ignore invalid (-1) values.
    """
    valid_data = df[df[column] != -1]
    mean = valid_data[column].mean()
    std = valid_data[column].std()

    z_scores = (valid_data[column] - mean) / std
    outliers = valid_data[np.abs(z_scores) > threshold]
    return outliers


#Loading data: Row 1-12,548
data = pd.read_csv("../MetData.csv", delimiter=',', nrows=12548)

#Drop unnecessary columns
drop_columns = [
    "Period", "Dynasty", "Reign", "Portfolio", "Subregion", "Locale", "Locus", "Excavation", "River",
    "Metadata Date", "Classification", "Rights and Reproduction", "County", "Object Wikidata URL","Region",
    "Artist Gender", "State", "Artist ULAN URL", "Artist Display Bio", "Artist Wikidata URL", "Dimensions",
    "Geography Type","Tags AAT URL", "Tags Wikidata URL", "Link Resource", "Credit Line", "Artist Prefix", "Tags", "Artist Alpha Sort"
]
data.drop(columns=drop_columns, inplace=True)

#Fill
data["Gallery Number"] = pd.to_numeric(data["Gallery Number"], errors='coerce').fillna(-1).astype(int)
data["AccessionYear"] = data["AccessionYear"].fillna(data["AccessionYear"].median()).astype(int)
data["Constituent ID"] = pd.to_numeric(data["Constituent ID"], errors='coerce').fillna(-1).astype(int)
data["Department"] = data["Department"].str.lower()
date_columns = ["Artist Begin Date", "Artist End Date", "Object Date", "Object Begin Date", "Object End Date"]

for col in date_columns:
    data[col] = data[col].astype(str).apply(extract_year)

for col in date_columns:
    data.loc[data[col] < 1000, col] = -1

#standardize
data["Culture"] = data["Culture"].apply(standardize_culture)
data["Artist Nationality"] = data["Artist Nationality"].apply(standardize)
data["Object Name"] = data["Object Name"].apply(standardize)
data["Title"] = data["Title"].apply(standardize)
data["Artist Role"] = data["Artist Role"].apply(standardize)
data["Artist Display Name"] = data["Artist Display Name"].apply(standardize)
data["Artist Suffix"] = data["Artist Suffix"].apply(standardize)
data["City"] = data["City"].apply(standardize)
data["Country"] = data["Country"].apply(standardize)
data["Repository"] = data["Repository"].apply(standardize)
data["Medium"] = data["Medium"].apply(standardize)

#changing boolean columns to 1 or 0
boolean_columns = ["Is Highlight", "Is Timeline Work", "Is Public Domain"]
data[boolean_columns] = data[boolean_columns].astype(int)

#this helps me find duplicates
data["Object Number"] = data["Object Number"].astype(str).apply(clean_object_number)
#dropping those duplicates, there are ~400+
data.drop_duplicates(subset=["Object Number"], keep="first", inplace=True)

for column in date_columns:
    outliers_iqr = detect_outliers_iqr(data, column)
    print(f"\nOutliers detected in {column} using IQR method: {len(outliers_iqr)}")
    print(outliers_iqr[["Object Number", column]].head())

    outliers_zscore = detect_outliers_zscore(data, column)
    print(f"\nOutliers detected in {column} using Z-score method: {len(outliers_zscore)}")
    print(outliers_zscore[["Object Number", column]].head())



print("\n\nChecking for duplicate rows:")
print(f"Total duplicate rows (entire dataset): {data.duplicated().sum()}")
print(f"Duplicate rows based on Object ID: {data.duplicated(subset=['Object ID']).sum()}")
print(f"Duplicate rows based on Object Number: {data.duplicated(subset=['Object Number']).sum()}")
print("\n\n")


data.to_csv("MetData_Cleaned.csv", index=False)