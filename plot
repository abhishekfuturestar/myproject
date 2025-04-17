# Filter out 'Unnamed' columns
df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Show the cleaned head (without unnamed columns)
print(df_clean.head())
