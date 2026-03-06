import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("groceries.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------------
# 2. Convert Rows into Transactions
# -------------------------------
# Drop 'Item(s)' column if present
data_items = data.drop(columns=['Item(s)'])

# Convert each row to list & remove NaN
transactions = data_items.apply(
    lambda row: row.dropna().tolist(),
    axis=1
).tolist()

print("\nSample Transactions:")
print(transactions[:5])

# -------------------------------
# 3. One-Hot Encoding
# -------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print("\nEncoded Data Shape:", df.shape)

# -------------------------------
# 4. Frequent Itemsets
# -------------------------------
frequent_itemsets = apriori(
    df,
    min_support=0.01,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

# -------------------------------
# 5. Association Rules
# -------------------------------
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']].head())

# -------------------------------
# 6. Top 10 Frequent Items
# -------------------------------
item_frequencies = df.sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(
    x=item_frequencies.head(10).values,
    y=item_frequencies.head(10).index
)

plt.title("Top 10 Frequent Items")
plt.xlabel("Frequency")
plt.ylabel("Items")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Confidence Heatmap
# -------------------------------
rules['antecedents_str'] = rules['antecedents'] \
                            .apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'] \
                            .apply(lambda x: ', '.join(list(x)))

top_ants = rules.groupby('antecedents_str')['support'] \
                .sum().nlargest(10).index
top_cons = rules.groupby('consequents_str')['support'] \
                .sum().nlargest(10).index

filtered = rules[
    (rules['antecedents_str'].isin(top_ants)) &
    (rules['consequents_str'].isin(top_cons))
]

heatmap_data = filtered.pivot_table(
    index='antecedents_str',
    columns='consequents_str',
    values='confidence'
)

plt.figure(figsize=(12,8))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Confidence'}
)

plt.title("Heatmap of Confidence for Top Rules")
plt.xlabel("Consequents")
plt.ylabel("Antecedents")
plt.tight_layout()
plt.show()

