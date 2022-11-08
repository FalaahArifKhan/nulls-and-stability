import numpy as np
import pandas as pd

def partition_by_group_intersectional(df, column_names, priv_values):
    priv = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    dis = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    return priv, dis

def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv)+len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis

def set_protected_groups(X_test, column_names, priv_values):
    priv_inter, dis_inter = partition_by_group_intersectional(X_test, column_names, priv_values)
    priv_1, dis_1 = partition_by_group_binary(X_test, column_names[0], priv_values[0])
    priv_2, dis_2 = partition_by_group_binary(X_test, column_names[1], priv_values[1])
    groups = {}
    return priv_inter, dis_inter, priv_1, dis_1, priv_2, dis_2

def set_protected_groups(X_test, column_names, priv_values):
    groups={}
    groups[column_names[0]+'_'+column_names[1]+'_priv'], groups[column_names[0]+'_'+column_names[1]+'_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_priv'], groups[column_names[0]+'_dis'] =  partition_by_group_binary(X_test, column_names[0], priv_values[0])
    groups[column_names[1]+'_priv'], groups[column_names[1]+'_dis'] =  partition_by_group_binary(X_test, column_names[1], priv_values[1])
    return groups

def get_fraction_of_nulls(df):
    return (df.shape[0] - df.dropna().shape[0])/ df.shape[0]


def generate_bootstrap(features, labels, bootstrap_size, with_replacement=True):
    bootstrap_index = np.random.choice(features.shape[0], size=int(bootstrap_size*features.shape[0]), replace=with_replacement)
    bootstrap_features = pd.DataFrame(features).iloc[bootstrap_index].values
    bootstrap_labels = pd.DataFrame(labels).iloc[bootstrap_index].values
    return bootstrap_features, bootstrap_labels

def UQ_by_boostrap(X_train, y_train, X_test, base_model, n_estimators, boostrap_size, with_replacement=True):
    '''
    Quantifying uncertainty of predictive model by constructing an ensemble from boostrapped samples
    '''
    predictions = {}
    ensemble = {}
    
    for m in range(n_estimators):
        model = base_model
        X_sample, y_sample = generate_bootstrap(X_train, y_train, boostrap_size, with_replacement)
        model.fit(X_sample, y_sample)
        predictions[m] = model.predict_proba(X_test)[:, 0]

        ensemble[m] = model
    return ensemble, predictions


def compute_label_stability(predicted_labels):
    '''
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1
    If the absolute difference is large, the label is more stable
    If the difference is exactly zero then it's extremely unstable --- equally likely to be classified as 0 or 1
    '''
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos
    return np.abs(count_pos - count_neg)/len(predicted_labels)

def confusion_matrix_metrics(confusion_matrix):
    metrics={}
    TN, FP, FN, TP = confusion_matrix.ravel()
    metrics['TPR'] = TP/(TP+FN)
    metrics['TNR'] = TN/(TN+FP)
    metrics['PPV'] = TP/(TP+FP)
    metrics['FNR'] = FN/(FN+TP)
    metrics['FPR'] = FP/(FP+TN)
    metrics['Accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    metrics['F1'] = (2*TP)/(2*TP+FP+FN)
    return metrics

def statistical_parity(y_preds, y_true):
    return y_preds.mean()/y_true.mean()



