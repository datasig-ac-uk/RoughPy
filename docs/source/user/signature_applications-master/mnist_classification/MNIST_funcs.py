import numpy as np
import esig
from esig import tosig as ts
import statistics
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import joblib
from joblib import Parallel, delayed 

def stream_normalise_mean_and_range(stream):
    """
    stream is a numpy array of any shape.

    The returned output is a copy of stream, retaining the same shape, but scaled
    to have mean 0 and coordinates/channels in [-1,1]
    """

    Q = stream.ptp(axis=0).shape
    E = np.empty(Q,dtype=float)
    for k in range(len(stream.ptp(axis=0))):
        if stream.ptp(axis=0)[k] == 0:
            E[k] = 1
        else:
            E[k] = stream.ptp(axis=0)[k]
    Y = (stream - stream.mean(axis=0))/E
    return Y

def stream_normalise_mean_and_std(stream):
    """
    stream can be a numpy array of any shape.

    The returned output is a copy of stream, retaining the same shape, but scaled
    to have mean 0 and standard deviation 1.0
    """

    Q = stream.std(axis=0).shape
    E = np.empty(Q,dtype=float)
    for k in range(len(stream.std(axis=0))):
        if stream.std(axis=0)[k] == 0:
            E[k] = 1
        else:
            E[k] = stream.std(axis=0)[k]
    Y = (stream - stream.mean(axis=0))/E
    return Y

def sig_scale_depth_ratio(sig, dim, depth, scalefactor):
    """
    sig is a numpy array corresponding to the signature of a stream, computed by
    esig, of dimension dim and truncated to level depth. The input scalefactor
    can be any float.

    The returned output is a numpy array of the same shape as sig. Each entry in
    the returned array is the corresponding entry in sig multiplied by r^k,
    where k is the level of sig to which the entry belongs. If the input depth
    is 0, an error message is returned since the scaling will always leave the
    level zero entry of a signature as 1.
    """

    if depth == 0:
        return print("Error: Depth 0 term of a signature is always 1 and will not be changed by scaling")
    else:
        e = []
        e.append(sig[0])
        for j in range(1, dim + 1):
            e.append( scalefactor*sig[j] )
        if depth >= 2:
            for k in range( dim + 1 , ts.sigdim( dim , 2) ):
                e.append( (scalefactor**2)*sig[k] )
            for i in range(2, depth):
                for a in range( ts.sigdim( dim , i ) , ts.sigdim( dim , i+1 ) ):
                    e.append( (scalefactor**(i+1))*sig[a] )
        E = np.array(e)
        return E

def mnist_train_data(data_directory, cpu_number):
    """
    Read the training data from the specified directory.

    Returns three lists; the first is a list of the streams formed by the 2D
    points data, the second is a list of streams of the 3D input data recording
    in the first two entries the change in x and y coordinates from the previous
    point and when the pen leaves the paper in the third entry. The fourth
    coordinate used to indicate when one figure ends and the next begins is
    ignored in this extraction. See the MNIST sequences project website for full
    details. The third returned list is a list of integer labels marking the
    number corresponding to each sequence.
    """

    def func_A(data_directory,k):
        return np.loadtxt('./{}/sequences/trainimg-{}-points.txt'.format(data_directory,k) , delimiter = ',' , skiprows = 1)[:-1,:]
    def func_B(data_directory,k):
        return np.loadtxt('./{}/sequences/trainimg-{}-inputdata.txt'.format(data_directory,k) , delimiter = ' ' , usecols = np.arange(0,3) )[:-1,:]
    def func_C(data_directory, k):
        return statistics.mode( np.argmax( np.loadtxt('./{}/sequences/trainimg-{}-targetdata.txt'.format(data_directory, k) , delimiter = ' ' ), axis=1) )
    A = Parallel(n_jobs = cpu_number)([ delayed(func_A)(data_directory, j) for j in range(60000) ])
    B = Parallel(n_jobs = cpu_number)([ delayed(func_B)(data_directory, j) for j in range(60000) ])
    C = Parallel(n_jobs = cpu_number)([ delayed(func_C)(data_directory, j) for j in range(60000) ])

    return A,B,C

def mnist_test_data(data_directory, cpu_number):
    """
    Read the training data from the specified directory.

    Returns three lists; the first is a list of the streams formed by the 2D
    points data, the second is a list of streams of the 3D input data recording
    in the first two entries the change in x and y coordinates from the previous
    point and when the pen leaves the paper in the third entry. The fourth
    coordinate used to indicate when one figure ends and the next begins is
    ignored in this extraction. See the MNIST sequences project website for full
    details. The third returned list is the integer labels marking the number
    corresponding to each sequence.
    """

    def func_A(data_directory, k):
        return np.loadtxt('./{}/sequences/testimg-{}-points.txt'.format(data_directory,k) , delimiter = ',' , skiprows = 1)[:-1,:]
    def func_B(data_directory, k):
        return np.loadtxt('./{}/sequences/testimg-{}-inputdata.txt'.format(data_directory,k) , delimiter = ' ' , usecols = np.arange(0,3) )[:-1,:]
    def func_C(data_directory, k):
        return statistics.mode( np.argmax( np.loadtxt('./{}/sequences/testimg-{}-targetdata.txt'.format(data_directory,k), delimiter = ' '), axis=1))
    A = Parallel(n_jobs=cpu_number)([ delayed(func_A)(data_directory, j) for j in range(10000) ])
    B = Parallel(n_jobs=cpu_number)([ delayed(func_B)(data_directory, j) for j in range(10000) ])
    C = Parallel(n_jobs=cpu_number)([ delayed(func_C)(data_directory, j) for j in range(10000) ])

    return A,B,C


def ridge_learn(scale_param, dim, depth, data, labels, CV=None, reg= [np.array((0.1,1,10))], cpu_number = 1):
    """
    scale_param is a float, dim and depth are positive integers, data is a list
    of numpy arrays with each array having the shape of an esig stream2sig
    output for a stream of dimension dim truncated to level depth, labels is a
    list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), reg is a numpy array of
    floats (its default is numpy.array((0.1,1.0,10.0))), and cpu_number is an
    integer (its default value is 1).
    
    The entries in the data list are scaled via the sig_scale_depth_ratio
    function, i.e. via sig_scale_depth_ratio(data, dim, depth,
    scalefactor=scale_param), and cpu_number number of cpus are used for
    parallelisation.
    
    Once scaled, a sklearn GridSearchCV is run with the model set to be
    RidgeClassifierCV(), the param_grid to be {'alphas':reg} and the
    cross-validation strategy to be determined by CV. 
    
    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed.
    
    The returned output is a list composed of the scale_param used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        ridge = linear_model.RidgeClassifierCV()
        tuned_params = {'alphas':reg}
        Q = Parallel(n_jobs=cpu_number)([ delayed(sig_scale_depth_ratio)(data[k] , dim , depth , scale_param) for k in range(len(data)) ])
        model = GridSearchCV(estimator=ridge, param_grid=tuned_params , cv=CV , n_jobs=cpu_number)
        model.fit(Q,labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_param, best_model, acc

def ridge_scale_learn(scale_parameters, dim, depth, data, labels, CV=None, reg = [np.array((0.1,1,10))], cpu_number_one = 1, cpu_number_two=1):
    """
    scale_parameters is a list of floats, dim and depth are positive integers,
    data is a list of numpy arrays with each array having the shape of an esig
    stream2sig output for a stream of dimension dim truncated to level depth,
    labels is a list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), reg is a numpy array of
    floats (its default is numpy.array((0.1,1.0,10.0))), and cpu_number_one and
    cpu_number_two are integers (default values are 1).
    
    The returned outputs are a list of the results of ridge_learn(r, dim, depth,
    data, labels, CV, reg) for each entry r in scale_parameters, and a separate
    recording of the output corresponding to the entry r in scale_parameters for
    which the best accuracy_score is achieved. The integer cpu_number_one
    determines how many cpus are used for parallelisation over the scalefactors,
    whilst cpu_number_two determines the number of cpus used for parallelisation
    during the call of the function ridge_learn for each scalefactor considered.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (cpu_number_one != 1 or cpu_number_two != 1):
        return print("Error: Tried to allocate more than the number of available cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)([ delayed(ridge_learn)(r, dim, depth, data, labels, CV, reg, cpu_number=cpu_number_two) for r in scale_parameters ])
        best = np.argmax( np.delete( results , 1, axis=1 ) , axis=0 )[1]
        return results, results[best]

def SVC_learn(scale_param, dim, depth, data, labels, CV=None, regC = [1.0], reg_gamma = ['scale'], reg_kernel = ['rbf'], cpu_number=1):
    """
    scale_param is a float, dim and depth are positive integers, data is a list
    of numpy arrays with each array having the shape of an esig stream2sig
    output for a stream of dimension dim truncated to level depth, labels is a
    list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (its deault is None), regC is a list of
    possible C inputs for a SVC model (default is [1.0]), reg_gamma is a list of
    possible gamma value strategies for a SVC model (default is ['scale']),
    reg_kernel is a list of possible kernels to be used in a SVC model (default
    is ['rbf']), and cpu_number is an integer controlling the number of cpus
    used for parallelisation (its default value is 1)
    
    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with scale_param as the scalefactor, i.e.
    sig_scale_depth_ratio(data, dim, depth, scalefactor = scale_param).
    
    Once scaled, a sklearn GridSearchCV is run with the model set to be
    SVC(random_state=42), the param_grid to be {'C':regC , 'gamma':reg_gamma ,
    'kernel':reg_kernel} and the cross-validation strategy to be determined by
    CV.
    
    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed. The returned output is a list composed of the
    scale_param used, the model selected during the GridSearch, and the
    accuracy_score achieved by the selected model.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        SVM = SVC(random_state=42)
        tuned_params = {'kernel':reg_kernel , 'C':regC , 'gamma':reg_gamma}
        Q = Parallel(n_jobs=cpu_number)([ delayed(sig_scale_depth_ratio)(data[k] , dim , depth , scale_param) for k in range(len(data)) ])
        model = GridSearchCV(estimator=SVM, param_grid=tuned_params , cv=CV , n_jobs=cpu_number)
        model.fit(Q,labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_param, best_model, acc

def SVC_scale_learn(scale_parameters, dim, depth, data, labels, CV=None, regC = [1.0], reg_gamma = ['scale'], reg_kernel =['rbf'], cpu_number_one=1, cpu_number_two=1):
    """
    scale_parameters is a list of floats, dim and depth are positive integers
    data is a list of numpy arrays with each array having the shape of an esig
    stream2sig output for a stream of dimension dim truncated to level depth,
    labels is a list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (its deault is None), regC is a list of
    possible C inputs for a SVC model (default is [1.0]), reg_gamma is a list of
    possible gamma value strategies for a SVC model (default is ['scale']),
    reg_kernel is a list of possible kernels to be used in a SVC model (default
    is ['rbf']), and    cpu_number_one and cpu_number_two are integers
    controlling the number of cpus used for parallelisation (default values are
    1). 
    
    The returned outputs are a list of the results of SVC_learn(r, dim, depth,
    data, labels, CV, regC, reg_gamma, reg_kernel) for each entry r in
    scale_parameters, and a separate recording of the output corresponding to
    the entry r in scale_parameters for which the best accuracy_score is
    achieved. 
    
    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scalefactors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    SVC_learn for each scalefactor considered.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (cpu_number_one != 1 or cpu_number_two != 1):
        return print("Error: Tried to allocate more than the available number of cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)([ delayed(SVC_learn)(r, dim, depth, data, labels, CV, regC, reg_gamma, reg_kernel, cpu_number=cpu_number_two) for r in scale_parameters ])
        best = np.argmax( np.delete( results , 1, axis=1 ) , axis=0 )[1]
        return results, results[best]

def logistic_learn(scale_param, dim, depth, data, labels, CV=None, regC = [5], no_iter=[100], cpu_number=1):
    """
    scale_parameters is a float, dim and depth are positive integers, data is a
    list of numpy arrays with each array having the shape of an esig stream2sig
    output for a stream of dimension dim truncated to level depth, labels is a
    list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), regC is a list of possible
    Cs inputs for a LogisticRegressionCV model (default is [5]), no_iter is a
    list of possible maximum number of iterations for a     LogisticRegressionCV
    model (default is [100]) and  cpu_number is an integer controlling the
    number of cpus used for parallelisation (its default value is 1). 
    
    The entries in the data list are scaled via the sig_scale_depth_ratio
    function using scale_param as the scalefactor, i.e via
    sig_scale_depth_ratio(data, dim, depth, scalefactor=scale_param).
    
    Once scaled, a sklearn GridSearchCV is run with the model set to be
    LogisticRegressionCV(random_state=42), the param_grid to be {'Cs':regC ,
    'max_iter':no_iter} and the cross-validation strategy to be determined by
    CV. 
    
    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed.
    
    The returned output is a list composed of the scale_param used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        LR = linear_model.LogisticRegressionCV(random_state=42)
        tuned_params = { 'Cs':regC , 'max_iter':no_iter }
        Q = Parallel(n_jobs=cpu_number)([ delayed(sig_scale_depth_ratio)(data[k] , dim , depth , scale_param) for k in range(len(data)) ])
        model = GridSearchCV(estimator=LR, param_grid=tuned_params , cv=CV , n_jobs=cpu_number)
        model.fit(Q,labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_param, best_model, acc

def logistic_scale_learn(scale_parameters, dim, depth, data, labels, CV=None, regC = [5], no_iter=[100], cpu_number_one=1, cpu_number_two=1):
    """
    scale_parameters is a list of floats, dim and depth are positive integers,
    data is a list of numpy arrays with each array having the shape of an esig
    stream2sig output for a stream of dimension dim truncated to level depth,
    labels is a list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), regC is a list of possible
    Cs inputs for a LogisticRegressionCV model (default is [5]), no_iter is a
    list of possible maximum number of iterations for a LogisticRegressionCV
    model (default is [100]), and cpu_number_one and cpu_number_two are integers
    controlling the number of cpus used for parallelisation (default values are
    1). 
    
    The returned outputs are a list of the results of logistic_learn(r, dim,
    depth, data, labels, CV, regC, no_iter) for each entry r in
    scale_parameters, and a separate recording of the output corresponding to
    the entry r in scale_parameters for which the best accuracy_score is
    achieved. 
    
    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scalefactors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    logistic_learn for each scalefactor considered.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (cpu_number_one != 1 or cpu_number_two != 1):
        return print("Error: Tried to allocate more than the available number of cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)([ delayed(logistic_learn)(r, dim, depth, data, labels, CV, regC, no_iter, cpu_number = cpu_number_two) for r in scale_parameters ])
        best = np.argmax( np.delete( results , 1 , axis=1 ) , axis=0)[1]
        return results, results[best]

def forest_learn(scale_param, dim, depth, data, labels, CV=None, reg_est = [10,100] , reg_feat = ['auto'], cpu_number=1):
    """
    scale_param is a float, dim and depth are positive integers, data is a list
    of numpy arrays with each array having the shape of an esig stream2sig
    output for a stream of dimension dim truncated to level depth, labels is a
    list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), reg_est is a list of
    possible n_estimators inputs for a RandomForestClassifier model (default is
    [10,100]), reg_feat is is a list of max_features strategies for a
    RandomForestClassifier model (default is ['auto']), and cpu_number is an
    integer controlling the number of cpus used for parallelisation (its default
    value is 1). 
    
    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with scale_param as the scalefactor, i.e
    sig_scale_depth_ratio(data, dim, depth, scalefactor=scale_param). 
    
    Once scaled, a sklearn GridSearchCV is run with the model set to be
    RFC(random_state=42), the param_grid to be {'max_features':reg_feat ,
    'n_estimators':reg_est} and the cross-validation strategy to be determined
    by CV. 
    
    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed. 
    
    The returned output is a list composed of the scale_param used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        forest = RFC(random_state=42, n_jobs=cpu_number)
        tuned_params = {'n_estimators':reg_est , 'max_features':reg_feat  }
        Q = Parallel(n_jobs=cpu_number)([ delayed(sig_scale_depth_ratio)(data[k] , dim , depth , scale_param) for k in range(len(data)) ])
        model = GridSearchCV(estimator=forest, param_grid=tuned_params , cv=CV , n_jobs=cpu_number)
        model.fit(Q,labels)
        best_model = model.best_estimator_ 
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_param, best_model, acc

def forest_scale_learn(scale_parameters, dim, depth, data, labels, CV=None, reg_est = [10,100] , reg_feat = ['auto'], cpu_number_one=1, cpu_number_two=1):
    """
    scale_parameters is a list of floats, dim and depth are positive integers,
    data is a list of numpy arrays with each array having the shape of an esig
    stream2sig output for a stream of dimension dim truncated to level depth,
    labels is a list, same length as data list, of integers, CV determines the
    cross-validation splitting strategy of a sklearn GridSearchCV and can be any
    of the allowed options for this (deault is None), reg_est is a list of
    possible n_estimators inputs for a RandomForestClassifier model (default is
    [10,100]), reg_feat is is a list of max_features strategies for a
    RandomForestClassifier model (default is ['auto']), and cpu_number_one and
    cpu_number_two are integers controlling the number of cpus used for
    parallelisation (default values are 1).

    The returned outputs are a list of the results of forest_learn(r, dim,
    depth, data, labels, CV, reg_est, reg_feat) for each entry r in
    scale_parameters, and a separate recording of the output corresponding to
    the entry r in scale_parameters for which the best accuracy_score is
    achieved. 
    
    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scalefactors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    forest_learn for each scalefactor considered.
    """

    if depth == 0:
        return print("Error: Depth 0 term of signature is always 1 and will not change under scaling")
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (cpu_number_one != 1 or cpu_number_two != 1):
        return print("Error: Tried to allocate more than the number of available cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)([ delayed(forest_learn)(r, dim, depth, data, labels, CV, reg_est, reg_feat, cpu_number=cpu_number_two) for r in scale_parameters ])
        best = np.argmax( np.delete( results , 1 , axis =1 ) , axis=0 )[1]
        return results, results[best]

def inv_reset_like_sum(stream):
    """
    The input stream is a numpy array. The returned output is a numpy array of
    the same shape, but where the n^th entry in the output is given by the
    component-wise sum of the first n entries in stream.
    """

    if isinstance( stream , np.ndarray ) == True:
        A = np.zeros( stream.shape , dtype=float )
        for k in range(len(A)):
            A[k] = np.sum( stream[:k+1] , axis=0 ).tolist()
        return A
    else:
        return print("Error: Input is not a numpy array")

def kmeans_fit(a, data):
    """
    a is an integer in [2, len(data) - 1] and data is a list of float-valued
    numpy arrays.

    The returned output is a KMeans cluster model, with n_clusters = a, fitted
    to data.
    """

    if a < 2 or a >= len(data):
        return print("Error: Number of clusters must be an integer in [{},{}]".format(2,len(data)-1))
    else:
        return KMeans(n_clusters=a, random_state=42).fit(data)

def kmeans_cluster_number(k_range, data, cpu_number=1):
    """
    k_range is a set of integers in [2, len(data) - 1], data is a list of
    float-valued numpy arrays and cpu_number is an integer controlling the
    number of cpus used for parallelisation (its default value is 1).
    
    For each integer j in k_range, a KMeans cluster model looking for j clusters
    is fitted to data. 
    
    The silhouette_score of each trained model is computed. 
    
    The returned output is the model achieving the highest silhouette_score and
    its associated silhouette_score.
    """

    if min(k_range) < 2 or max(k_range) >= len(data):
        return print("Error: k_range must be a set of integers within [{},{}]".format(2, len(data) - 1))
    else:
        kmeans_per_k = Parallel(n_jobs=cpu_number)([ delayed(kmeans_fit)(a,data) for a in k_range ])
        silhouette_scores = [ silhouette_score(data, model.labels_) for model in kmeans_per_k ]
        best_index = np.argmax(silhouette_scores)
        best_k = k_range[best_index]
        best_score = silhouette_scores[best_index]
        return best_k , best_score

def kmeans_percentile_propagate(cluster_num , percent, data, labels):
    """
    cluster_num is an integer in [2, len(data)-1], percent is a float in
    [0,100], data is a list of float-valued numpy arrays and labels is a list,
    of same length as data, of integers.
    
    A KMeans model with cluster_num clusters is used to fit_transform data.
    
    A representative centroid element from each cluster is assigned its
    corresponding label from the labels list.
    
    This label is then propagated to the percent% of the instances in the
    cluster that are closest to the clusters centroid.
    
    The returned output is a list of these chosen instances and a list of the
    corresponding labels for these instances.
    """

    if percent > 100 or percent < 0:
        return print("Error: percent must be in [{},{}]".format(0,100))
    if cluster_num > len(data) - 1 or cluster_num < 2:
        return print("Error: cluster_num must be an integer in [{},{}]".format(2,len(data) - 1))
    else:
        kmeans = KMeans(n_clusters = cluster_num , random_state=42)
        data_dist = kmeans.fit_transform(data)
        data_cluster_dist = data_dist[ np.arange( len(data) ) , kmeans.labels_ ]
        for i in range(cluster_num):
            in_cluster = (kmeans.labels_ == i)
            cluster_dist = data_cluster_dist[in_cluster]
            cutoff_distance = np.percentile( cluster_dist , percent )
            above_cutoff = (data_cluster_dist > cutoff_distance)
            data_cluster_dist[in_cluster & above_cutoff] = -1
        partially_propagated = (data_cluster_dist != -1)
        data_pp , labels_pp = [] , []
        for z in range(len(partially_propagated)):
            if partially_propagated[z] == True:
                data_pp.append(data[z])
                labels_pp.append(labels[z])
        return data_pp , labels_pp

def model_performance(B, dim, depth, data, labels, cpu_number):
    """
    B is a list with B[0] being a float and B[1] a trained classifier that is
    already fitted to (data, labels), dim and depth are positive integers, data
    is a list of float-valued numpy arrays, with each array having the shape of
    an esig stream2sig output for a stream of dimension dim truncated to level
    depth, labels is a list (same length as data) of integers, and cpu_number is
    an integer controlling the number of cpus used for parallelisation.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with B[0] as the scalefactor, i.e.  sig_scale_depth_ratio(data,
    dim, depth, scalefactor = B[0]). 
    
    The model B[1] is used to make predictions using the rescaled data, and the
    accuracy_score, the confusion matrix and a normalised confusion_matrix
    (focusing only on the incorrect classifications) of these predictions
    compared to the labels are computed. 
    
    The returned outputs are the scalefactor B[0] used, the model B[1] used, the
    accuracy_score achieved, the confusion_matrix and the normalised
    confusion_matrix highlighting the errors.
    """

    scaled_data = Parallel(n_jobs=cpu_number)([ delayed(sig_scale_depth_ratio)(a, dim, depth, B[0]) for a in data ])
    preds = B[1].predict(scaled_data)
    acc = accuracy_score(preds, labels)

    CM = confusion_matrix(preds, labels)
    row_sums = CM.sum(axis=1, keepdims=True)
    for i in range(len(row_sums)):
        if row_sums[i] == 0:
            row_sums[i] = 1
    CM_norm = CM/row_sums
    np.fill_diagonal(CM_norm,0)
    
    return B[0], B[1], acc, CM, CM_norm

