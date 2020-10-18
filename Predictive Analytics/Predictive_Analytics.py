# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import linear_model 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random




def import_dataset(fileName):
    #Importing dataset
    dataset=pd.read_csv(fileName)
    dataset.columns = [*dataset.columns[:-1], 'labels']
    return dataset

def MinMaxScaling(dataset):
    #Scaling the dataset
    scaler = MinMaxScaler()
    scaler.fit(dataset.drop('labels',axis=1))
    scaled_features = scaler.transform(dataset.drop('labels',axis=1))
    df = pd.DataFrame(scaled_features,columns=dataset.columns[:-1])
    df['labels'] = dataset['labels']
    return df

def train_test_split(dataset):
    #Splitting into Train and test datasets
    ncols=dataset.shape[1]
    train = dataset.sample(frac=0.8, random_state=0)
    test = dataset.drop(train.index)
    X_train,Y_train=train.iloc[:, 0:ncols-1].values,train.iloc[:, -1].values
    X_test,Y_test=test.iloc[:, 0:ncols-1].values,test.iloc[:, -1].values
    return X_train,Y_train,X_test,Y_test


def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    return (y_true==y_pred).mean()

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    macro_recall=np.mean(recall)
    return macro_recall

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    macro_precision=np.mean(precision)
    return macro_precision


def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss = 0
    for i in range(len(Clusters)):
        centroid = np.mean(Clusters[0], 0)
        dis = (np.sum((Clusters[0] - centroid) ** 2, axis=1))
        wcss += np.sum(dis)
    return wcss

def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    step_1=(y_true*num_of_classes)+y_pred
    step_2=np.histogram(step_1, bins=np.arange((num_of_classes+1),((num_of_classes**2)+num_of_classes+2)))[0]
    confusion_matrix=step_2.reshape((num_of_classes,num_of_classes))
    return confusion_matrix

def KNN(X_train,X_test,Y_train,K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    distances = np.sqrt(np.sum(X_train**2, axis=1) + np.sum(X_test**2, axis=1)[:, np.newaxis] - 2 * np.dot(X_test, X_train.T))
    sortdistancesidx = distances.argsort()
    Kindices=sortdistancesidx[:,0:K]
    neighbours=np.take(Y_train, Kindices)
    neighbours_df=pd.DataFrame(neighbours)
    Y_pred=neighbours_df.mode(axis=1)[0].to_frame().astype(int)
    Y_pred=Y_pred.values
    Y_pred=Y_pred.reshape((len(Y_pred),))
    return Y_pred
    
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """

    def check_pure(data):
        labels=data[:,-1]
        uniq_classes=np.unique(labels)
        if len(uniq_classes)==1:
            return True
        else:
            return False
    
    def Classify(data):
        labels=data[:,-1]
        uniq_classes, counts_uniq_classes=np.unique(labels, return_counts=True)
        index=counts_uniq_classes.argmax()
        classification=uniq_classes[index]
        return classification
    
    def fetch_potential_splits(data,random_subspace):
        potential_splits={}
        ncols=data.shape[1]
        col_indices=list(range(ncols-1))
        if random_subspace:
            col_indices=random.sample(population=col_indices,k=random_subspace)
        for col_index in col_indices:
            potential_splits[col_index]=[]
            values=data[:,col_index]
            unique_values=np.unique(values)
            length=len(unique_values) 
            unique_values1=unique_values[0:length-1:]
            unique_values2=unique_values[1:length:]
            potential_splits[col_index]=(unique_values1+unique_values2)/2  
        return potential_splits
    
    def split_data(data,split_column,split_value):
        split_col_value=data[:,split_column]
        split1=data[split_col_value<=split_value]
        split2=data[split_col_value>split_value]
        return split1,split2
    
    
    def gini_impurity(data):
        labels=data[:, -1]
        _,counts=np.unique(labels,return_counts=True)   
        p=counts/counts.sum()
        gini_impurity=1-sum(p**2)
        return gini_impurity
    
    def calc_info_gain(current_gain,split1,split2):
        n_data_points=len(split1)+len(split2)
        if n_data_points>0:
            p_split1=len(split1)/n_data_points
            p_split2=len(split2)/n_data_points
        else:
            return 0       
        info_gain=current_gain-(p_split1*gini_impurity(split1)+p_split2*gini_impurity(split2))
        return info_gain
    
    
    def determine_best_split(data,potential_splits):
        curr_gain=gini_impurity(data)
        best_gain=0
        for i in potential_splits.keys():
            potential_splits[i]=np.mean(potential_splits[i])
        for col_index in potential_splits:
            split1,split2=split_data(data,col_index,potential_splits[col_index])
            child_info_gain=calc_info_gain(curr_gain,split1,split2)
            if child_info_gain>=best_gain:
                best_gain=child_info_gain
                best_split_column=col_index
                best_split_value=potential_splits[col_index]
        return best_split_column,best_split_value
    
    
    def decision_tree(train,counter=0,min_samples=2,max_depth=8,random_subspace=None):
        if counter==0:
            data=train.values
        else:
            data=train

        if (check_pure(data)) or (len(data)<min_samples) or (counter==max_depth):
            classification=Classify(data)
            return classification

        else:
            counter+=1
            potential_splits=fetch_potential_splits(data,random_subspace)
            split_column,split_value=determine_best_split(data,potential_splits)
            split1,split2=split_data(data,split_column,split_value)

            question="{} <= {}".format(split_column,split_value)
            sub_tree={question:[]}
            
            yes_answer=decision_tree(split1,counter,min_samples,max_depth,random_subspace)
            no_answer=decision_tree(split2,counter,min_samples,max_depth,random_subspace)
            if yes_answer==no_answer:
                sub_tree=yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
            return sub_tree
        
    def classify_example(example,tree):
        question=list(tree.keys())[0]
        values=question.split()
        column=int(values[0])
        split_value=values[2]
        if example[column]<=float(split_value):
            answer=tree[question][0]
        else:
            answer=tree[question][1]
        #base case
        if not isinstance(answer, dict):
            return int(answer)
        else:
            return classify_example(example,answer)
    
    def classify_all(test,tree):
        ydf = pd.DataFrame(columns=['pred'])
        ydf=test.apply(classify_example,axis=1,args=(tree,))
        return ydf
    
    def random_forest_algorithm(train, n_trees, n_features, dt_max_depth):
        forest = []
        for i in range(n_trees):
            df_bootstrapped = train.sample(frac=0.4,replace=True,random_state=i)
            tree = decision_tree(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
            forest.append(tree)
            print("Tree_"+str(i)+" is completed")
        return forest
    
    def random_forest_predictions(test_df, forest):
        dict_predictions = {}
        for i in range(len(forest)):
            column_name = "tree_"+str(i)
            predictions = classify_all(test_df, tree=forest[i])
            dict_predictions[column_name] = list(predictions)
        df_predictions=pd.DataFrame.from_dict(dict_predictions)
        return df_predictions
    
    train_df=pd.DataFrame(X_train)
    train_df[49]=pd.DataFrame(Y_train)
    test_df=pd.DataFrame(X_test)
    
    forest=random_forest_algorithm(train_df,n_trees=13,n_features=7,dt_max_depth=30)
    rf_predictions=random_forest_predictions(test_df,forest)
    rf_predictions = (rf_predictions.mode(axis=1)[0]).to_frame().astype(int)
    rf_predictions=rf_predictions.to_numpy()
    rf_predictions=rf_predictions.reshape((len(rf_predictions),))
    return rf_predictions
    
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    means = np.mean(X_train.T, axis=1)
    centered_matrix = X_train - means
    covariance = np.cov(centered_matrix.T)
    values, vectors = np.linalg.eig(covariance)
    vectors = vectors[:, :N]
    pca = vectors.T.dot(centered_matrix.T)    
    return pca.T


def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    def initialize_clusters(points, k):
        return points.iloc[np.random.randint(points.shape[0], size=k)]
    
    def get_distances(centroid, points):
        return np.linalg.norm(points - centroid, axis=1)

    k = N
    X_train = pd.DataFrame(data=X_train)
    maxiter = 10
    centroids = initialize_clusters(X_train, k)
    classes = np.zeros(X_train.shape[0], dtype=np.float64)
    distances = np.zeros([X_train.shape[0], k], dtype=np.float64)
    
    for i in range(maxiter):
        for j in range(len(centroids)):
            distances[:, j] = get_distances(centroids.iloc[j], X_train)
    
        classes = np.argmin(distances, axis=1)
        for c in range(k):
            centroids.iloc[c] = np.mean(X_train[classes == c], 0)
    res = []    
    for i in range(k):
        temp=X_train[classes == i]
        res.append(temp.values)
    return res
    

def SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    def sklearnsSVM(X_train,Y_train,X_test):
        svm_model_linear = svm.SVC(kernel = 'linear')
        svm_model_linear.fit(X_train, Y_train) 
        Y_pred = svm_model_linear.predict(X_test)
        return Y_pred
  
    def sklearnsLogisticRegression(X_train,Y_train,X_test):
        model = linear_model.LogisticRegression() 
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        return Y_pred
        
    def sklearnsDecisiontree(X_train,Y_train,X_test):
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        return Y_pred
    
    def sklearnsKNN(X_train,Y_train,X_test):
        knn = KNeighborsClassifier()
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        return Y_pred  
    
    def SklearnSLAccuracies(Y_test,predictionlist):
        algorithms=["SVM","Logistic Regression","Decision Tree","KNN"]
        Accuracies={}
        for algorithm,pred in zip(algorithms,predictionlist):
            Accuracies[algorithm]=Accuracy(Y_test,pred)
        return Accuracies
    
    svm_predictions=sklearnsSVM(X_train,Y_train,X_test)
    lr_predictions=sklearnsLogisticRegression(X_train,Y_train,X_test)
    dt_predictions=sklearnsDecisiontree(X_train,Y_train,X_test)
    knn_predictions=sklearnsKNN(X_train,Y_train,X_test)
    predictionlist=[svm_predictions,lr_predictions,dt_predictions,knn_predictions]
    print("Accuracies of each model")
    print(SklearnSLAccuracies(Y_test,predictionlist))
    return [svm_predictions,lr_predictions,dt_predictions,knn_predictions]

        
def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    model = VotingClassifier(estimators=[('svm', svm.SVC(kernel = 'linear')), ('lr', linear_model.LogisticRegression() ), ('dt', DecisionTreeClassifier()),('knn', KNeighborsClassifier())], voting='hard')
    model = model.fit(X_train, Y_train)
    Y_pred=model.predict(X_test)
    print("Voting Classifier Accuracy: "+str(Accuracy(Y_test,Y_pred)))
    return Y_pred

  
"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""


def plot_confusion_matrix(confusionmatrixList,algorithms=["SVM","Logistic Regression","Decision Tree","KNN","Voting Classifier"]):
    
    def cartesian_coord(array1,array2):
        grid = np.meshgrid(array1,array2)        
        coord_list = [entry.ravel() for entry in grid]
        points = np.vstack(coord_list).T
        return points
    

    fig, axs = plt.subplots(2,3, figsize=(20, 10))
    plt.subplots_adjust(top = 0.95, bottom=0.04, hspace=0.4, wspace=0.4)
    n_algorithms=len(algorithms)
    print(n_algorithms)
    k=0
    for i in range(2):
        for j in range(3): 
            if k<n_algorithms:
                axs[i][j].set_title("confusion matrix for "+algorithms[k], pad=20)
                axs[i][j].imshow(confusionmatrixList[k], interpolation='nearest', cmap=plt.cm.Greens)
                axs[i][j].set_xticks(np.arange(1,12), [str(i) for i in range(1,12)])
                axs[i][j].set_yticks(np.arange(1,12), [str(i) for i in range(1,12)])
                tick_marks = np.arange(-0.5,11.5)
                classes=[i for i in range(12)]
                axs[i][j].set_xticks(tick_marks,classes)
                axs[i][j].set_yticks(tick_marks,classes)
                thresh = confusionmatrixList[k].max() / 2.
                for s, t in cartesian_coord(range(confusionmatrixList[k].shape[0]), range(confusionmatrixList[k].shape[1])):
                    axs[i][j].text(t, s, format(confusionmatrixList[k][s, t], 'd'),
                             horizontalalignment="center",
                             color="white" if confusionmatrixList[k][s, t] > thresh else "black")
                axs[i][j].set_ylabel('True label')
                axs[i][j].set_xlabel('Predicted label')
                plt.tight_layout()
                k+=1
            else:
                axs[i][j].set_visible(False)
    plt.rcParams['figure.dpi'] = 200
    plt.show();




def GridSearch(X_train,Y_train,X_test,Y_test):
    models = {
        'SVM': svm.SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'KNN':KNeighborsClassifier()
    }
    
    
    param_grid = {
        'SVM': [
                 {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                 {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.0001]}
                ],      
        'DecisionTreeClassifier':  { 
                'criterion' : ['gini', 'entropy'],
                'max_depth' : [4,6,8,12] },
        'KNN': { 'n_neighbors': [3,5,11,19],
                'weights':['uniform','distance'],
                'metric':['euclidean','manhattan'] }
    }
    # Instantiate the grid search model
    grid_search_svm = GridSearchCV(estimator = models['SVM'], param_grid = param_grid['SVM'], n_jobs = -1, verbose = 1)
    grid_search_dt = GridSearchCV(estimator = models['DecisionTreeClassifier'], param_grid = param_grid['DecisionTreeClassifier'], 
                              n_jobs = -1, verbose = 1)
    grid_search_knn = GridSearchCV(estimator = models['KNN'], param_grid = param_grid['KNN'], 
                              n_jobs = -1, verbose = 1)
    
    grid_search_svm.fit(X_train,Y_train)
    grid_search_dt.fit(X_train,Y_train)
    grid_search_knn.fit(X_train,Y_train)
    
    
    # Print the tuned parameters and score 
    print("Tuned SVM Parameters: {}".format(grid_search_svm.best_params_)) 
    print("Best estimator is {}".format(grid_search_svm.best_estimator_)) 
    
    print("Tuned Decision Tree Parameters: {}".format(grid_search_dt.best_params_)) 
    print("Best estimator is {}".format(grid_search_dt.best_estimator_)) 
    
    print("Tuned KNN Parameters: {}".format(grid_search_knn.best_params_)) 
    print("Best estimator is {}".format(grid_search_knn.best_estimator_)) 
    
    grid_search_svm_pred=grid_search_svm.predict(X_test)
    grid_search_dt_pred=grid_search_dt.predict(X_test)
    grid_search_knn_pred=grid_search_knn.predict(X_test)
    
    print("SVM Classification report")
    print(classification_report(Y_test, grid_search_svm_pred)) 
    
    print("Decision Tree Classification report")
    print(classification_report(Y_test, grid_search_dt_pred)) 
    
    print("KNN Classification report")
    print(classification_report(Y_test, grid_search_knn_pred)) 
    
    
    def svm_gridseatch_plot(grid_search_model,param_grid,ax):
        scores_svm = grid_search_model.cv_results_['mean_test_score'][4:13]
        scores_svm = np.array(scores_svm).reshape(len(param_grid['SVM'][1]['C']), len(param_grid['SVM'][1]['gamma']))
        
        for i, j in enumerate(param_grid['SVM'][1]['C']):
            ax.plot(param_grid['SVM'][1]['gamma'], scores_svm[i], label='C: ' + str(j))
        ax.legend()
        ax.set_title('Mean test scores for SVM with linear kernel')
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Mean score')
    
    
    def dt_gridseatch_plot(grid_search_model,param_grid,ax):
        scores_dt = grid_search_model.cv_results_['mean_test_score']
        scores_dt = np.array(scores_dt).reshape(len(param_grid['DecisionTreeClassifier']['criterion']), len(param_grid['DecisionTreeClassifier']['max_depth']))
        
        for k, l in enumerate(param_grid['DecisionTreeClassifier']['criterion']):
            ax.plot(param_grid['DecisionTreeClassifier']['max_depth'], scores_dt[k], label='criterion: ' + str(l))
        ax.legend()
        ax.set_title('Mean test scores for Decision Tree')
        ax.set_xlabel('Max depth')
        ax.set_ylabel('Mean score')
    
    def knn_gridseatch_plot(grid_search_model,param_grid,ax):
        selectedscores=[]
        scores_knn = grid_search_model.cv_results_['mean_test_score']
        criteria=grid_search_model.cv_results_['params']
        for i in range(len(criteria)):
            if criteria[i]['weights']=='uniform':
                selectedscores.append(scores_knn[i])
        
        scores_knn = np.array(selectedscores).reshape(len(param_grid['KNN']['metric']), len(param_grid['KNN']['n_neighbors']))
        
        for m, n in enumerate(param_grid['KNN']['metric']):
            ax.plot(param_grid['KNN']['n_neighbors'], scores_knn[m], label='metric: ' + str(n))
        ax.legend()
        ax.set_title('Mean test scores for KNN')
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('Mean score')
    
        
    _, ax = plt.subplots(2, 2) 
    plt.subplots_adjust(top = 0.95, bottom=0.09, hspace=0.6, wspace=0.4)
    svm_gridseatch_plot(grid_search_svm,param_grid,ax[0,0])
    dt_gridseatch_plot(grid_search_dt,param_grid,ax[0,1])
    knn_gridseatch_plot(grid_search_knn,param_grid,ax[1,0])
    ax[1,1].set_visible(False)


num_of_classes=len(pd.read_csv("data.csv")['48'].unique())

def main():
    dataset=import_dataset("data.csv")
    scaled_dataset=MinMaxScaling(dataset)
    X_train,Y_train,X_test,Y_test=train_test_split(scaled_dataset)
    knn_pred=KNN(X_train,X_test,Y_train,5)
    print("KNN")
    print("Accuracy: "+str(Accuracy(Y_test,knn_pred)))
    print("Recall: "+str(Recall(Y_test,knn_pred)))
    print("Precision: "+str(Precision(Y_test,knn_pred)))
    print("KNN Confusion Matrix")
    print(ConfusionMatrix(Y_test,knn_pred))
    k_means=Kmeans(X_train,10)
    print("WCSS : "+str(WCSS(k_means)))
    rf_pred=RandomForest(X_train,Y_train,X_test)
    print("Random Forest")
    print("Accuracy: "+str(Accuracy(Y_test,rf_pred)))
    print("Recall: "+str(Recall(Y_test,rf_pred)))
    print("Precision: "+str(Precision(Y_test,rf_pred)))
    print("Random forest Confusion Matrix")
    print(ConfusionMatrix(Y_test,rf_pred))
    print("PCA Data")
    print(PCA(X_train,10))
    sklearnpredictions=SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test)
    SklearnVotingClassifierpreds=SklearnVotingClassifier(X_train,Y_train,X_test,Y_test)
    prediction_list=sklearnpredictions+[SklearnVotingClassifierpreds]
    cm_list=[]
    for k in prediction_list:
        cm_list.append(ConfusionMatrix(Y_test,k))     
    plot_confusion_matrix(cm_list)
    GridSearch(X_train,Y_train,X_test,Y_test)

if __name__== "__main__":
    main()
