import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from botocore.client import Config
import pandas as pd
import boto3

year = input('please enter a year')
quater = input('please enter a quater')

def path_compiler():
    path = './Q{}{}/historical_data1_time_Q{}{}.txt'.format(quater,year,quater,year)
    if not os.path.isfile(path):
        raise Exception("File does not exists!")
    else:
        return path;

def read_txt_file(path, col_header):
	df = pd.read_table(path, delimiter='|', header = None, names = col_header)
	return df

def dropCol(dataset, cutoff = 0.3):
    for col_name in columns_header:
        n = len(dataset)
        value_count = dataset[col_name].count()
        if(value_count/n) < cutoff:
            dataset.drop(col_name, axis = 1, inplace = True)
    return dataset
  
def toDatetime(dataset):
    dataset["MONTHLY_REPORTING_PERIOD"] = pd.to_datetime(dataset["MONTHLY_REPORTING_PERIOD"], format = "%Y%m")
    dataset['Year'] = pd.DatetimeIndex(dataset["MONTHLY_REPORTING_PERIOD"]).year
    dataset['Month'] = pd.DatetimeIndex(dataset["MONTHLY_REPORTING_PERIOD"]).month
    dataset.drop(["MONTHLY_REPORTING_PERIOD"], axis = 1, inplace = True)
    return dataset

def categorical_to_numerical(dataset):
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dataset[col] = dataset[col].astype('category')
    vec = DictVectorizer(sparse=False, dtype=int)
    dc = dataset.to_dict('records')
    result = vec.fit_transform(dc)
    return result

def divide_x_y(dataframe):
    Y = dataframe['CURRENT_LOAN_DELINQUENCY_STATUS']
    dataframe.drop(['CURRENT_LOAN_DELINQUENCY_STATUS'], axis = 1, inplace = True)
    df_Y = pd.DataFrame(Y)
    return dataframe, df_Y

def pre_poccessing(df):
    df = dropCol(df)
    df = toDatetime(df)
    x,y = divide_x_y(df)
    x.drop(['LOAN_SEQUENCE_NUMBER'], axis = 1, inplace = True)
    y = y.replace(to_replace=r'.*R.*', value = '-1', regex = True)
    y = y['CURRENT_LOAN_DELINQUENCY_STATUS'].values.astype('int32')
    x = categorical_to_numerical(x)
    return x, y

def draw_roc_curve(model, x_test, y_test, y_train):
    unique_y = np.unique(y_train)
    y_binarize = label_binarize(y_test, classes = unique_y)
    n_classes = y_binarize.shape[1]
    y_score = model.decision_function(x_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarize[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_binarize.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
    plt.figure(figsize = [20,20])
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='red', linestyle=':', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_confusion_matrix(cm, classes ,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def main():
    df_train = read_txt_file(columns_header)