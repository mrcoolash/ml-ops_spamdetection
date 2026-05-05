import pandas as pd

# Load raw data
df = pd.read_csv("data/spamraw.csv")

# Rename columns
df = df.rename(columns={
    "type": "label",
    "text": "text"
})

# Convert labels: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Save cleaned data
df.to_csv("data/data.csv", index=False)

print("Preprocessing done. Saved as data/data.csv")