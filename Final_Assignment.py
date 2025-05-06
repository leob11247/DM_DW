import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Data
Retail_Data = pd.read_excel("/Users/leobaltazar/Desktop/Data Mining/Online Retail.xlsx")


print("Original Data Shape:", Retail_Data.shape) #Displaying Data Shape before any cleaning

Retail_Data.drop_duplicates(inplace=True) #Removing Duplicates

print("After Dropping Duplicates:", Retail_Data.shape) #Displaying Shape after dropping duplicates

#Dropping Blank Rows
Retail_Data.dropna(subset=['CustomerID'], inplace=True)
#Properly Formatting Invoice Date
Retail_Data['InvoiceDate'] = pd.to_datetime(Retail_Data['InvoiceDate'])
#Removing rows with negative or zero Quantity or UnitPrice
Retail_Data = Retail_Data[(Retail_Data['Quantity'] > 0) & (Retail_Data['UnitPrice'] > 0)]
# Removes canceled transactions (InvoiceNo starts with 'C')
Retail_Data = Retail_Data[~Retail_Data["InvoiceNo"].astype(str).str.startswith("C")]
Retail_Data['Description'] = Retail_Data['Description'].str.strip()
#Convert CustomerID to integer
Retail_Data['CustomerID'] = Retail_Data['CustomerID'].astype(int)

# Remove rows where Description contains "?" or placeholder words
Retail_Data = Retail_Data[~Retail_Data["Description"].str.contains(r'\?', regex=True, na=False)]
print("Cleaned Data Shape:", Retail_Data.shape) #cleaning done, displayed


#When did the customer last buy?
Retail_Data_recency = Retail_Data.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
Retail_Data_recency.columns = ['CustomerID', 'LastPurchaseDate']
recent_date = Retail_Data_recency['LastPurchaseDate'].max() # finds the most recent date among all customers.
Retail_Data_recency['Recency'] = Retail_Data_recency['LastPurchaseDate'].apply(lambda x: (recent_date - x).days) # tells us how many days have passed since they last bought something
Retail_Data_recency.head()

# Frequency:  How often do they buy?
frequency = Retail_Data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index() #count how many unique invoices each customer has
frequency.columns = ['CustomerID', 'Frequency']

# Monetary: How much do they spend?
monetary = Retail_Data.copy() #makes a copy of the dataset.
monetary['TotalAmount'] = monetary['Quantity'] * monetary['UnitPrice'] # creates a new column. TotalAmount = Quantity × UnitPrice
monetary = monetary.groupby('CustomerID')['TotalAmount'].sum().reset_index()
monetary.columns = ['CustomerID', 'Monetary']
#groups by customer and sums up how much they’ve spent in total.

# Merge RFM features
rfm = Retail_Data_recency.merge(frequency, on='CustomerID').merge(monetary, on='CustomerID')
#Combines all 3 features (Recency, Frequency, Monetary) into one DataFrame using the common key: CustomerID.

# Label: has the customer bought more than once?
rfm['BoughtAgain'] = (rfm['Frequency'] > 1).astype(int)

# Prepare features and target
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['BoughtAgain']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rfm.to_excel("RFM_Predictions.xlsx", index=False)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
