import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/audit.csv")
df = df[~df["corrupt"]].drop_duplicates(subset="sha256")  # drop corrupt + exact dupes

train, temp = train_test_split(df, test_size=0.3, stratify=df["class"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["class"], random_state=42)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
print(len(train), len(val), len(test))