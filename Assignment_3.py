import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#This is for association rule assignment
#Leo Baltazar

Movies_Data = pd.read_csv("/Users/leobaltazar/Desktop/Data Mining/Movies.tsv", delimiter='\t')
Ratings_Data = pd.read_csv("/Users/leobaltazar/Desktop/Data Mining/Ratings.tsv", delimiter='\t')

Movies_Data["genres"] = Movies_Data["genres"].apply(lambda x: x.split("|"))

Ratings_Data["rating"] = pd.to_numeric(Ratings_Data["rating"]).astype(int)
high_ratings = Ratings_Data[Ratings_Data["rating"] >= 4]

merged_data = pd.merge(high_ratings, Movies_Data, on="movieId", how="inner")

transactions = merged_data.groupby("userId")["genres"].apply(lambda x: list(set(sum(x, []))))
transactions = transactions.tolist()

encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(encoded_data, columns=encoder.columns_)

frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)
print(frequent_itemsets.head(3))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

