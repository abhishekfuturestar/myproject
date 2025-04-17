# This will give you a DataFrame with only columns that have NO NaN values
df_no_nan_cols = df.loc[:, df.notna().all()]

print("Columns without any NaN values:")
print(df_no_nan_cols.columns)

# If you want to see the filtered DataFrame:
print(df_no_nan_cols)
