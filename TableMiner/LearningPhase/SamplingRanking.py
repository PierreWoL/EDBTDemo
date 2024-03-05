from sklearn.feature_extraction.text import CountVectorizer


# Redefine the function to reorder rows based on preference scores
def reorder_dataframe_rows(df, ne_column):
    """
    Reorders rows in a DataFrame based on preference scores derived from the bag-of-words model
    of the context around a specified NE-column.

    Args:
    - df (DataFrame): The DataFrame to be processed.
    - ne_column (str): The name of the NE-column based on which the reordering is performed.

    Returns:
    - DataFrame: The re-ordered DataFrame.
    """
    # Initialize CountVectorizer and preference scores dictionary
    vectorizer = CountVectorizer()
    preference_scores = {}

    # Iterate over unique NE-column values to calculate preference scores
    for ne_value in df[ne_column].unique():
        # Combine context for rows with the same NE-column value into a single document
        context_texts = df[df[ne_column] == ne_value].drop(columns=[ne_column]).astype(str).apply(' '.join, axis=1)
        combined_context = " ".join(context_texts)
        # Fit and transform the context to a bag-of-words model
        bow = vectorizer.fit_transform([combined_context])
        # Assign preference score based on the number of unique words
        preference_scores[ne_value] = bow.shape[1]

    # Create a score column in the DataFrame to sort by
    df['Preference_Score'] = df[ne_column].apply(lambda x: preference_scores[x])

    # Sort the DataFrame based on the preference score and drop the score column
    df_sorted = df.sort_values(by='Preference_Score', ascending=False).drop(columns=['Preference_Score'])

    return df_sorted

