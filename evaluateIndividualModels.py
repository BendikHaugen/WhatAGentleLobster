import pandas as pd
from latextResults.evaluateResults import create_latex_table, wrap_latex_table

df = pd.read_csv("latextResults/results.csv")
"""
Split window size from name and sort in ascending order
"""
df[['Model_name', 'Window size']] = df['model name'].str.rsplit('_', n=1, expand=True)
df['Window size'] = pd.to_numeric(df['Window size'].str[2:])
df['model name'] = df['Model_name'].str.replace('_ws', '')
df.drop("Model_name", axis=1, inplace=True)
df.rename(columns={"model name": "Model name"}, inplace=True)
df = df.sort_values(["Model name", "Window size"], ascending=True)
# split the 'model name' column into 'Model_name' and 'Feature vector'
df_joarve = df[df["Model name"].str.contains('joarve')]
df = df.drop(df_joarve.index)
df[['Model name', 'Feature vector']] = df['Model name'].str.split('_', n=1, expand=True)
feature_mapping = lambda s: "any claw to any keypoint" if s == "kp2kp" else "any claw to wireframe" if s == "kp2wf" else "both claws to wireframe" if s == "2kp2wf" else "Bounding box baseline" if s == "bbonly" else "Distance between centers" if s =="centerdist" else "IoS" if s =="ios" else "IoU" if s =="iou" else "keypoints baseline" if s =="kponly" else "All bounding boc features" if s =="allbbfeats" else s
model_mapping = lambda s: "LogReg" if s == "LogRegCV" else "Ridge" if s == "RidgeCV" else s
df['Feature vector'] = df['Feature vector'].apply(feature_mapping)
df['Model name'] = df['Model name'].apply(model_mapping)
cols = list(df.columns)
cols.insert(1, cols.pop(cols.index('Window size')))
cols.insert(1, cols.pop(cols.index('Feature vector')))
df = df[cols]
df = df.set_index(['Model name', 'Feature vector', 'Window size'])

import pandas as pd

def top_rows(df, column_name, n):
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' does not exist in the dataframe.")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError("'n' should be a positive integer.")
        
    return df.nlargest(n, column_name)

def top_rows_by_feature_vector(df, feature_vector_value, column_name, n):
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' does not exist in the dataframe.")

    if not isinstance(n, int) or n < 1:
        raise ValueError("'n' should be a positive integer.")

    sub_df = df[df.index.get_level_values('Feature vector') == feature_vector_value]
    return sub_df.nlargest(n, column_name)

def top_rows_by_threshold(df, sort_column, threshold_column, threshold, n):
    if sort_column not in df.columns or threshold_column not in df.columns:
        raise ValueError("One or both specified columns do not exist in the dataframe.")
    if not isinstance(n, int) or n < 1:
        raise ValueError("'n' should be a positive integer.")

    return df[df[threshold_column] > threshold].nlargest(n, sort_column)



def rows_with_index_containing(df, index_name, substring):
    if index_name not in df.index.names:
        raise ValueError(f"'{index_name}' does not exist in the dataframe indices.")

    return df[df.index.get_level_values(index_name).str.contains(substring)]


def bottom_rows(df, column_name, n):
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' does not exist in the dataframe.")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError("'n' should be a positive integer.")
        
    return df.nsmallest(n, column_name)

def bottom_rows_exclude_models(df, column_name, n, exclude_string):
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' does not exist in the dataframe.")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError("'n' should be a positive integer.")
        
    if 'Model name' not in df.index.names:
        raise ValueError("'Model name' does not exist in the dataframe index.")

    filtered_df = df[~df.index.get_level_values('Model name').str.contains(exclude_string)]
    return filtered_df.nsmallest(n, column_name)



df_lstm = rows_with_index_containing(df, "Model name", "LSTM")
df_rnn = rows_with_index_containing(df, "Model name", "RNN")
df_attention = rows_with_index_containing(df, "Model name", "Attention")
df_mlp = rows_with_index_containing(df, "Model name", "MLP")

'''
Først snakk om de modellene som gjorde det best
'''
top_auc = top_rows(df, "Val AUC", 5)
top_f1 = top_rows(df, "Val F1", 5)
top_auc_train = top_rows(df, "Train AUC", 5)
top_f1_train = top_rows(df, "Train F1", 5)

"""
Så de som gjorde det best innenfor hver ANN
"""
top_f1_lstm = top_rows(df_lstm, "Val F1", 5)
top_f1_rnn = top_rows(df_rnn, "Val F1", 5)
top_f1_attention = top_rows(df_attention, "Val F1", 5)
top_f1_mlp = top_rows(df_mlp, "Val F1", 5)


top_within_threshold = top_rows_by_threshold(df, "Val F1", "AVG FPS", 30, 5)
top_within_threshold_rnn = top_rows_by_threshold(df_rnn, "Val F1", "AVG FPS", 30, 5)
top_within_threshold_lstm = top_rows_by_threshold(df_lstm, "Val F1", "AVG FPS", 30, 5)
top_within_threshold_attention = top_rows_by_threshold(df_attention, "Val F1", "AVG FPS", 30, 5)

print(top_within_threshold)
print(top_within_threshold_rnn)
print(top_within_threshold_lstm)
print(top_within_threshold_attention)

tables = [
    top_auc, 
    top_f1,
    top_auc_train,
    top_f1_train,
    top_f1_lstm,
    top_f1_rnn,
    top_f1_attention,
    top_f1_mlp,
    top_within_threshold,
    top_within_threshold_rnn,
    top_within_threshold_lstm,
    top_within_threshold_attention,
]

worst_auc = bottom_rows(df, "Val AUC", 10)
worst_f1 = bottom_rows(df, "Val F1", 10)

not_rnn = bottom_rows_exclude_models(df, "Val AUC", 10, "RNN")


#with open(f'allModels.txt', 'w') as file:
#
#    for table in tables:
#        latex_table = create_latex_table(table)
#        latex_table = wrap_latex_table(latex_table)
#        file.write(latex_table)
#        file.write("\n")
#        file.write("*" * 100)
#        file.write("\n")

sucky_tables = [worst_auc, worst_f1, not_rnn]
with open(f'sucky.txt', 'w') as file:

    for table in sucky_tables:
        latex_table = create_latex_table(table)
        latex_table = wrap_latex_table(latex_table)
        file.write(latex_table)
        file.write("\n")
        file.write("*" * 100)
        file.write("\n")
