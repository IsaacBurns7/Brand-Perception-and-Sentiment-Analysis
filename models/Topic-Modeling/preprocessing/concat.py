import pandas as pd

# Load the files
df1 = pd.read_csv('../data/ModLewis_test.csv')
df2 = pd.read_csv('../data/ModLewis_train.csv')

# Stack them on top of each other
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the result
combined_df.to_csv('../data/reuters.csv', index=False)