import math, time, pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def print_well(well_data, figname):
    fig, axs = plt.subplots(1, 20, figsize=(14, 4), sharey=True)

    for ic, col in enumerate(sorted(DATA_COLUMNS)):
        axs[ic].plot(well_data[col], well_data['DEPTH_MD'])
        axs[ic].set_xlabel(col)

    axs[0].set_ylim(well_data['DEPTH_MD'].values[-1], well_data['DEPTH_MD'].values[0])
    fig.savefig(figname, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()

def print_histogram(well_data, figname):
    fig, ax = plt.subplots(5, 4, figsize=(20, 20))

    for id, col in enumerate(sorted(DATA_COLUMNS)):
        hist, bins = np.histogram(well_data[col].dropna(), bins='auto')
        ax[id // 4, id % 4].plot(bins[:hist.size], hist / np.sum(hist))
        ax[id // 4, id % 4].set_xlabel(col)

    fig.savefig(figname, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()

def load_and_group():
    global DATA_COLUMNS

    data = pd.read_csv('train_part_00.csv', sep=';')
    data = data.append(pd.concat((pd.read_csv(f, sep=';', names=data.columns) for f in ['train_part_01.csv', 'train_part_02.csv', 'train_part_03.csv'])))

    DATA_COLUMNS = set(data.columns) - set(['DEPTH_MD', 'FORCE_2020_LITHOFACIES_LITHOLOGY',
                                            'FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL', 'GROUP', 'FORMATION',
                                            'X_LOC', 'Y_LOC', 'Z_LOC'])

    print_histogram(data, 'histogram before anything (total %d items).png' % data.size)

    # rounds depth to 5 in 5 meters, change this as you want
    data['DEPTH_MD'] = data.DEPTH_MD.apply(lambda x: 5 * round(x / 5))

    # groups by depth, rock type and confidence, "gathering" data through their average
    grouped = data.groupby(['DEPTH_MD','FORCE_2020_LITHOFACIES_LITHOLOGY','FORCE_2020_LITHOFACIES_CONFIDENCE']).mean()

    #all wells as one
    print_well(grouped.reset_index(), 'all wells as one - raw data.png')
    print_histogram(grouped, 'histogram after grouping in 5 meters (total %d items).png' % grouped.size)

    return grouped

def fill(grouped):
    # testing different fillers
    '''
    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = SimpleImputer(strategy="most_frequent").fit_transform(grouped_filled)
    print_histogram(grouped_filled, 'histogram filled most_frequent.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled most_frequent.png')

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = SimpleImputer(strategy="constant").fit_transform(grouped_filled)
    print_histogram(grouped_filled, 'histogram filled constant.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled constant.png')

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = SimpleImputer(strategy="mean").fit_transform(grouped_filled)
    print_histogram(grouped_filled, 'histogram filled mean.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled mean.png')

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = IterativeImputer(max_iter=50, random_state=0).fit_transform(grouped_filled.values)
    print_histogram(grouped_filled, 'histogram filled IterativeImputer.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled IterativeImputer.png')

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = KNNImputer(n_neighbors=5, weights="uniform").fit_transform(grouped_filled.values)
    print_histogram(grouped_filled, 'histogram filled KNNImputer n=5.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled KNNImputer n=5.png')

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = KNNImputer(n_neighbors=50, weights="uniform").fit_transform(grouped_filled.values)
    print_histogram(grouped_filled, 'histogram filled KNNImputer n=50.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled KNNImputer n=50.png')'''

    grouped_filled = grouped.copy()
    grouped_filled[grouped_filled.columns] = KNNImputer(n_neighbors=500, weights="uniform").fit_transform(grouped_filled.values)
    print_histogram(grouped_filled, 'histogram filled KNNImputer n=500.png')
    print_well(grouped_filled.reset_index(), 'all wells together - filled KNNImputer n=500.png')

    return grouped_filled.reset_index()

def normalize(data):
    for col in sorted(set(DATA_COLUMNS) - set(['DTS', 'MUDWEIGHT', 'RXO', 'DCAL'])):
        scaler = PowerTransformer()
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    print_histogram(data, 'histogram normalized PowerTransformer.png')

    for col in ['DCAL', 'DRHO', 'MUDWEIGHT', 'ROPA', 'RXO', 'SP', 'DTS', 'RHOB']:
        scaler = QuantileTransformer(output_distribution='normal')
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    print_histogram(data, 'histogram normalized PowerTransformer -> QuantileTransformer.png')

    for col in DATA_COLUMNS:
        scaler = MinMaxScaler()
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    print_histogram(data, 'histogram normalized PowerTransformer -> QuantileTransformer -> MinMax.png')

    return data

def print_helper(y_true, y_pred, f):
    #remove unewanted data
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.iloc[:, :-1]
    df.drop(df.index[[3,4]], inplace=True)

    print(df.to_html(float_format="%.2f", decimal=','), file=f)
    print(file=f)

def find_best_mlp_classifier_params(X_train, X_test, y_train, y_test):
    with open('results mlp_classifier.html', 'w') as f:
        for s in ['lbfgs', 'sgd', 'adam']:
            start = time.time()

            parameters = {'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (5,5), (10,10), (20,20), (5,10,20), (20,10,5)],
                          'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                          'random_state': [1],
                          'max_iter': [10000],
                          'solver': [s]}

            clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_macro', n_jobs=-1, cv=5)
            clf.fit(X_train, y_train.ravel())
            y_true, y_pred = y_test, clf.predict(X_test)

            print(classification_report(y_true, y_pred))
            print(clf.best_params_)
            print("total time for", s, time.time() - start)

            print_helper(y_true, y_pred, f)
            f.flush()

#TODO implement this after finding the best parameters using find_best_mlp_classifier_params
def mlp_classifier(X_train, X_test, y_train, y_test):
    with open('results.html', 'w') as f:
        clf = MLPClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        print('mlp_classifier results', file=f)
        print(s, clf.best_params_, file=f)
        print_helper(y_true, y_pred, f)

def logistic_regression(X_train, X_test, y_train, y_test):
    with open('results.html', 'a') as f:
        clf = LogisticRegression(random_state=0, max_iter=10000, verbose=1, class_weight='balanced')
        clf.fit(X_train, y_train, X_train['FORCE_2020_LITHOFACIES_CONFIDENCE'].map({1:1000, 2:100, 3:1}).values)

        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        print('logistic regression results', file=f)
        print_helper(y_true, y_pred, f)

def svc(X_train, X_test, y_train, y_test):
    with open('results.html', 'a') as f:
        clf = SVC(gamma='auto', random_state=0, max_iter=10000, verbose=1, class_weight='balanced')
        clf.fit(X_train, y_train, X_train['FORCE_2020_LITHOFACIES_CONFIDENCE'].map({1:20, 2:10, 3:1}).values)

        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        print('svc results', file=f)
        print_helper(y_true, y_pred, f)

print('loading and grouping')
grouped = load_and_group()

print('filling empty values')
grouped_filled = fill(grouped)

print('normalizing')
normalized = normalize(grouped_filled)

y = grouped_filled['FORCE_2020_LITHOFACIES_LITHOLOGY']
y = y.map({30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11})

X = normalized
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('finding best mlp parameters')
find_best_mlp_classifier_params(X_train[DATA_COLUMNS], X_test[DATA_COLUMNS], y_train, y_test)

#print('running mlp classifier')
#mlp_classifier(X_train[DATA_COLUMNS], X_test[DATA_COLUMNS], y_train, y_test)

print('running logistic regression')
logistic_regression(X_train, X_test, y_train, y_test)

print('running svc')
svc(X_train, X_test, y_train, y_test)
