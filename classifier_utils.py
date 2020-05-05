import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from datetime import datetime
import itertools
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def train_model(name, model, X, y, categorical_features = [], pca_n_components=0, save=False):

    # Pipeline for sklearn-based one-hot encoder
    one_hotter = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Nan')),
        ('scaler',OneHotEncoder(dtype=int, categories='auto', handle_unknown='ignore',sparse=True))
    ])
    # Column transformen if needed
    column_trans = ColumnTransformer(
        [
            ('one_hotter',one_hotter, categorical_features)
        ]
        ,remainder='passthrough'
    )
    # Model steps
    steps = [
        ('transformer', column_trans),
        ('scaler', StandardScaler(copy=True, with_mean=False, with_std=True)),
        ('pca', PCA(pca_n_components) if pca_n_components !=0 else None),
        ('model', model)
    ]

    pipe = Pipeline(steps)
    pipe.fit(X,y)
    
    if save:
        fname='%s_%s.joblib' % (name, datetime.now().strftime("%Y%m%d"))
        print('Save model to: %s'%fname)
        joblib.dump(pipe, fname)
        
    return pipe

def cross_validate_model(model, X, y, 
                         scoring=['f1', 'precision', 'recall', 'roc_auc'], 
                         cv=12, n_jobs=-1, verbose=True):
    
    scores = cross_validate(model, 
                        X, y, 
                        scoring=scoring,
                        cv=cv, n_jobs=n_jobs, 
                        verbose=verbose,
                        return_train_score=False)

    #print(scores)

    #sorted(scores.keys())
    dd={}
    
    for key, val in scores.items():
        if key in ['fit_time', 'score_time']:
            continue
        #print('{:>30}: {:>6.5f} +/- {:.5f}'.format(key, np.mean(val), np.std(val)) )
        name = " ".join(key.split('_')[1:]).capitalize()
        
        dd[name] = {'value' : np.mean(val), 'error' : np.std(val)}
        
    return  pd.DataFrame(dd)

def plot_roc(model, X_test ,y_test, n_classes=0):
    
    from sklearn.metrics import roc_curve, auc
    
    """
    Target scores, can either be probability estimates 
    of the positive class, confidence values, or 
    non-thresholded measure of decisions (as returned 
    by “decision_function” on some classifiers).
    """
    try:
        y_score = model.decision_function(X_test)
    except Exception as e:
        y_score = model.predict_proba(X_test)[:,1]
    
    
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    #plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()

def plot_confusion_matrix(model, X_test ,y_test,
                          classes=[0,1],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float')
        cm_norm=np.zeros((2,2))
        np.fill_diagonal(cm_norm, np.diagonal(cm) / np.sum(cm,axis=1))
        np.fill_diagonal(cm_norm[::-1], np.diagonal(cm[::-1]) / np.sum(cm,axis=0))
        cm=cm_norm
        
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def feature_importance(coef, names, top_n=10, verbose=False, plot=True):
    
    #importances = model.feature_importances_

    
    
    #std = np.std([tree.feature_importances_ for tree in model.estimators_],
    #             axis=0)
    indices = np.argsort(coef)[::-1][:top_n]
    
    if verbose:
    
        # Print the feature ranking
        print("Feature ranking:")
    
        for f in range(len(indices)):
            print("{:>2d}. {:>15}: {:.5f}".format(f + 1, names[indices[f]], coef[indices[f]]))
        
    if plot:
        
        # Plot the feature importances of the forest
        #plt.figure(figsize=(5,10))
        plt.title("Feature importances")
        plt.barh(range(len(indices)), coef[indices][::-1], align="center")
        #plt.barh(range(X.shape[1]), importances[indices][::-1],
        #         xerr=std[indices][::-1], align="center")
        plt.yticks(range(len(indices)), names[indices][::-1])
        #plt.xlim([-0.001, 1.1])
        #plt.show()

def plot_proba(model, X, y, calib=False, show_class = 1):
    
    if calib:
        calib = CalibratedClassifierCV(model, cv='prefit')
    
        calib.fit(X, y)
    
        proba=calib.predict_proba(X)
    
    else:
        proba=model.predict_proba(X)
    
    if show_class == 0:
        #sns.kdeplot(proba[y==0,0], shade=True, clip=(0,1), cut=0, color="r", label='True class')
        #sns.kdeplot(proba[y==0,1], shade=True, clip=(0,1), cut=0, color="b", label='Wrong class')
        plt.hist(proba[y==0,0], bins=20, range=(0,1), alpha=0.5, density=1, color="r", label='True class')
        plt.hist(proba[y==0,1], bins=20, range=(0,1), alpha=0.5, density=1, color="b", label='Wrong class')
        plt.title('Classification probability')
    elif show_class == 1:
        #sns.kdeplot(proba[y==1,1], shade=True, clip=(0,1), cut=0, color="r", label='True class')
        #sns.kdeplot(proba[y==1,0], shade=True, clip=(0,1), cut=0, color="b", label='Wrong class')
        w = np.ones_like(proba[y==1,1])/float(len(proba[y==1,1]))
        plt.hist(proba[y==1,1], bins=20, range=(0,1), alpha=0.3, density=False, weights=w, color="r", label='True class')
        w = np.ones_like(proba[y==1,0])/float(len(proba[y==1,1]))
        plt.hist(proba[y==1,0], bins=20, range=(0,1), alpha=0.3, density=False, weights=w, color="b", label='Wrong class')

        plt.title('Classification probability')
    plt.legend()

def plot_proba_calibration(model, X_test, y_test):

    # predict probabilities
    probs = model.predict_proba(X_test)[:,1]
    # reliability diagram
    fop, mpv = calibration_curve(y_test, probs, n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positive')
    plt.title('Model reliability')
    #plt.show()

def plot_precision_recall_curve(model, X_test, y_test):
    
    """
    Target scores, can either be probability estimates 
    of the positive class, confidence values, or 
    non-thresholded measure of decisions (as returned 
    by “decision_function” on some classifiers).
    """
    try:
        y_score = model.decision_function(X_test)
    except Exception as e:
        y_score = model.predict_proba(X_test)[:,1]

    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)

    #print('Average precision-recall score: {0:0.2f}'.format(
    #  average_precision))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from inspect import signature

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    
    plt.step(recall, precision, color='b', alpha=1,  where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    #plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
    #          average_precision))
    


def plot_report(name, model, X_test, y_test, figsize = (15,20)):

    plt.figure(figsize = figsize)

    plt.suptitle('Classification report: %s' % name, y=1.05, fontsize=20)

    plt.subplot(421)
    plot_proba_calibration(model, X_test, y_test)

    plt.subplot(422)
    plot_roc(model, X_test ,y_test)

    plt.subplot(423)
    plot_confusion_matrix(model, X_test ,y_test, normalize=True)

    plt.subplot(424)
    plot_confusion_matrix(model, X_test ,y_test, normalize=False)

    plt.subplot(425)
    plot_proba(model, X_test, y_test, calib=False)

    plt.subplot(426)
    plot_precision_recall_curve(model, X_test, y_test)

    plt.subplot(427)
    #if hasattr(model.named_steps['model'], 'coef_'):
    #    feature_importance(model.named_steps['model'].coef_[0], X_test.columns, top_n=10)
    #elif hasattr(model.named_steps['model'], 'feature_importances_'):
    #    feature_importance(model.named_steps['model'].feature_importances_, X_test.columns, top_n=10)
    if hasattr(model.named_steps['model'], 'coef_'):
        feature_imp = model.named_steps['model'].coef_[0]
    elif hasattr(model.named_steps['model'], 'feature_importances_'):
        feature_imp = model.named_steps['model'].feature_importances_
    feature_nam = np.array(['Feature %d' % i for i in range(len(feature_imp))])
    feature_importance(feature_imp, feature_nam, top_n=10)

    plt.tight_layout()

def reg_plot_report(name, model, X_test, y_test, figsize = (10,5)):

    plt.figure(figsize = figsize)
    plt.suptitle('Regression report: %s' % name, y=1.05, fontsize=20)

    plt.subplot(121)
    y_pred = model.predict(X_test)
    pd.Series(y_test.values - y_pred).plot.hist(bins=100, logy=True)
    plt.ylabel('')
    plt.xlabel('Truth - Prediction')
    
    plt.subplot(122)
    if hasattr(model.named_steps['model'], 'coef_'):
        feature_imp = model.named_steps['model'].coef_
    elif hasattr(model.named_steps['model'], 'feature_importances_'):
        feature_imp = model.named_steps['model'].feature_importances_
    
    feature_nam = np.array(['Feature %d' % i for i in range(len(feature_imp))])
    feature_importance(feature_imp, feature_nam, top_n=10)    
    
    plt.tight_layout()