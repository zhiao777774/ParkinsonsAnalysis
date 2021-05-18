import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
import graphviz
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap

from preprocess import feature_names, x_train, x_test, y_train, y_test


def plot_decision_regions(X, y, title=None, label_names=None, *, save_fig=False, file_name=None):
    # setup marker generator and color map
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl if not label_names else label_names[cl])

    plt.legend()

    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(file_name)


def correlation():
    _df = pd.read_csv('./datasets/parkinsons.data')
    _x = _df.drop(['name'], 1)
    _x = _x.drop(['status'], 1)

    corr1 = pd.DataFrame()
    col_names = ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',	'Jitter:DDP','NHR']
    for col in col_names:
        corr1[col] = _x[col]
    print(corr1.corr())

    sns.heatmap(corr1.corr(),annot=True, fmt='.2f', xticklabels=col_names, yticklabels=col_names)
    plt.title('Pearson correlation coefficients between the vocal features of \n MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, \n MDVP:PPQ, Jitter:DDP, and NHR.\n')
    plt.savefig('./assets/image/correlation_heatmap_1.png')
    plt.show()
    
    corr2 = pd.DataFrame()
    col_names = ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA']
    for col in col_names:
        corr2[col] = _x[col]
    print(corr2.corr())

    sns.heatmap(corr2.corr(),annot=True, fmt='.2f', xticklabels=col_names, yticklabels=col_names)
    plt.title('Pearson correlation coefficients between the vocal features of \n MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, \n Shimmer:APQ5, and Shimmer:DDA.\n')
    plt.savefig('./assets/image/correlation_heatmap_2.png')
    plt.show()

    corr3 = pd.DataFrame()
    col_names = ['spread1','spread2', 'PPE']
    for col in col_names:
        corr3[col] = _x[col]
    print(corr3.corr())
    
    sns.heatmap(corr3.corr(),annot=True, fmt='.2f', xticklabels=col_names, yticklabels=col_names)
    plt.title('Pearson correlation coefficients between the vocal features of \n spread1, spread2, and PPE.\n')
    plt.savefig('./assets/image/correlation_heatmap_3.png')
    plt.show()

def decisionTree(output_file=False, *, file_name=None, file_format='png'):
    decision_tree = DecisionTreeClassifier()

    parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 30),
        'min_samples_leaf': range(5, 50, 5),
        'min_samples_split': range(20, 200, 20)
    }
    grid = GridSearchCV(decision_tree, param_grid=parameters, cv=5)
    grid.fit(x_train, y_train)
    best_clf = grid.best_estimator_

    
    print("Decision Tree's feature names: ", feature_names)
    print("Decision Tree's feature importances: ")
    for n, p in zip(feature_names, best_clf.feature_importances_):
        print(f'{n}: {round(p, 2)}')
    print("Decision Tree's predict (test data): ", best_clf.predict(x_test))
    print("Decision Tree's Accuracy (test data): ", best_clf.score(x_test, y_test))

    if output_file:
        # Draw Tree
        dot_data = export_graphviz(best_clf,
                                   feature_names=feature_names,
                                   class_names=["Parkinson's", 'healthy'],
                                   filled=True)
        graph = graphviz.Source(dot_data)
        graph.format = file_format
        graph.render(file_name)

    return best_clf


def adjust_k():
    k_range = range(1, 31)
    knn_pipe = Pipeline([('scaler', StandardScaler()),
                         ('knn', KNeighborsClassifier(n_jobs=-1))])
    knn_params = {'knn__n_neighbors': k_range}

    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5,
                            n_jobs=-1, verbose=True)

    scaler = StandardScaler()
    knn_grid.fit(scaler.fit_transform(x_train), y_train)

    scores = []
    ks = []
    for k, s in zip(knn_grid.cv_results_['params'],
                    knn_grid.cv_results_['mean_test_score']):
        scores.append(s)
        ks.append(k['knn__n_neighbors'])

    dt = pd.DataFrame({'k': ks, 'accuracy': scores})
    display(dt)

    font = FontProperties(
        fname=r'D:\\桌面\\Python\\assets\\fonts\\Microsoft-JhengHei.ttf', size=15)
    plt.plot(k_range, scores, 'g-o', label='accuracy')
    plt.title('最佳k值搜索 [1, 30]', fontproperties=font)
    plt.xlabel("k value", fontsize=12, labelpad=15)
    plt.ylabel("accuracy (%)", fontsize=12, labelpad=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/image/best_k_value(1 ~ 30).png')
    plt.show()

    print('KNN最佳參數: ', knn_grid.best_params_, knn_grid.best_score_)


def knn(plot=True):
    knn = KNeighborsClassifier(n_neighbors=1)

    scaler = StandardScaler()
    x_train_scalered = scaler.fit_transform(x_train)
    print(x_train, sep='\n')
    print(x_train_scalered)
    knn.fit(x_train_scalered, y_train)

    plt.figure(figsize=(8, 6))
    sns.distplot(x_train)
    plt.title("Distribution of original data (training data)", y=1.015)
    plt.savefig('./assets/image/original distribution.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(x_train_scalered)
    plt.title("Distribution of transformed (training data)", y=1.015)
    plt.savefig('./assets/image/standard scaler distribution.png')
    plt.show()

    knn_pred = knn.predict(scaler.transform(x_test))
    print("KNN's Accuracy: ", accuracy_score(y_test, knn_pred))

    if plot:
        plot_decision_regions(scaler.transform(
            x_test), y_test, title='KNN (k=1)', label_names={1: "Parkinson's", 0: 'healthy'},
            save_fig=True, file_name='./assets/image/KNN.png')

    return knn


def logisticRegression():
    clf = LogisticRegression(max_iter=10000)

    parameters = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2']
    }

    grid = GridSearchCV(clf, param_grid=parameters, cv=5)
    grid.fit(x_train, y_train)

    print(grid.score(x_test, y_test))
    print(grid.predict(x_test))

    return grid


def logisticRegressionWithStandard():
    clf = LogisticRegression(max_iter=10000)

    parameters = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2']
    }

    scaler = StandardScaler()
    x_train_scalered = scaler.fit_transform(x_train)
    x_test_scalered = scaler.fit_transform(x_test)

    grid = GridSearchCV(clf, param_grid=parameters, cv=5)
    grid.fit(x_train_scalered, y_train)

    print(grid.score(x_test_scalered, y_test))
    print(grid.predict(x_test_scalered))

    return grid


def log_reg_model_credit_card_applications_predict(x, log_reg):
    return 1 / (1 + np.exp(-(log_reg.intercept_ + log_reg.coef_ * x)))


def logisticRegressionCV():
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scalered = scaler.transform(x_train)
    x_test_scalered = scaler.transform(x_test)

    clf = LogisticRegressionCV(cv=5, max_iter=10000)
    clf.fit(x_train_scalered, y_train)

    print("Logstic Regression with CV's Accuracy: ",
          clf.score(x_test_scalered, y_test))
    print('Logstic Regression\'s intercept', clf.intercept_[0])
    print('Logstic Regression\'s coefficient')
    for n, c in zip(feature_names, clf.coef_[0]):
        print(f'{n}: {c}')

    
    cf_matrix = confusion_matrix(y_test, clf.predict(scaler.transform(x_test)))
    print(cf_matrix)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    ticks = ['healthy', 'Parkinson\'s']
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=ticks, yticklabels=ticks)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('./assets/image/logistic regression\'s confusion matrix.png')
    
    '''
    logit_model=sm.Logit(y_train, x_train_scalered)
    logit_model.exog_names[:] = feature_names
    result=logit_model.fit()

    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    '''

    '''
    fpr,tpr,threshold = roc_curve(y_test, clf.predict(x_test_scalered)) ###計算真正率和假正率
    roc_auc = auc(fpr,tpr) ###計算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='r',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率為橫座標，真正率為縱座標做曲線
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''

    '''
    plt.figure(figsize=(10, 10))
    plt.scatter(x=x_train_scalered, y=log_reg_model_credit_card_applications_predict(
        x_train_scalered, clf), c='blueviolet', alpha=0.5)

    orange_rect = patches.Rectangle(
        (-2.7, 0.5), 9, 0.5, linewidth=1, edgecolor='orange', facecolor='navajowhite', alpha=0.5)
    plt.gca().add_patch(orange_rect)
    blue_rect = patches.Rectangle(
        (-2.7, 0), 9, 0.5, linewidth=1, edgecolor='blue', facecolor='steelblue', alpha=0.5)
    plt.gca().add_patch(blue_rect)
    plt.title("Predictions on Training Data", y=1.015, fontsize=20)
    plt.xlabel("feature scaled", labelpad=14)
    plt.ylabel("probability of a prediction being class 1 (accepted)", labelpad=14)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(x=x_test_scalered, y=log_reg_model_credit_card_applications_predict(
        x_test_scalered, clf), c='blueviolet', alpha=0.5)

    orange_rect = patches.Rectangle(
        (-2.7, 0.5), 9, 0.5, linewidth=1, edgecolor='orange', facecolor='navajowhite', alpha=0.5)
    plt.gca().add_patch(orange_rect)
    blue_rect = patches.Rectangle(
        (-2.7, 0), 9, 0.5, linewidth=1, edgecolor='blue', facecolor='steelblue', alpha=0.5)
    plt.gca().add_patch(blue_rect)
    plt.title("Predictions on Test Data", y=1.015, fontsize=20)
    plt.xlabel("feature scaled", labelpad=14)
    plt.ylabel("probability of a prediction being class 1 (accepted)", labelpad=14)
    plt.show()
    '''

    return clf


if __name__ == '__main__':
    # dt = decisionTree(output_file=True,
    #             file_name='./assets/image/decision_tree', file_format='png')
    # adjust_k()
    # knn_clf = knn(plot=True)
    # lr = logisticRegression()
    # lr_with_s = logisticRegressionWithStandard
    # lr_cv = logisticRegressionCV()
    correlation()
