############### IMPORT PACKAGES ##################

import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter

#Used for naive bayes decoder
try:
    import statsmodels.api as sm
except ImportError:
    print("\nWARNING: statsmodels is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    import math
except ImportError:
    print("\nWARNING: math is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
except ImportError:
    print("\nWARNING: scipy is not installed. You will be unable to use the Naive Bayes Decoder")
    pass



#Import scikit-learn (sklearn) if it is installed
try:
    from sklearn import linear_model #For Wiener Filter and Wiener Cascade
    from sklearn.svm import SVR #For support vector regression (SVR)
    from sklearn.svm import SVC #For support vector classification (SVM)
except ImportError:
    print("\nWARNING: scikit-learn is not installed. You will be unable to use the Wiener Filter or Wiener Cascade Decoders")
    pass

#Import XGBoost if the package is installed
try:
    import xgboost as xgb #For xgboost
except ImportError:
    print("\nWARNING: Xgboost package is not installed. You will be unable to use the xgboost decoder")
    pass

#Import functions for Keras if Keras is installed
#Note that Keras has many more built-in functions that I have not imported because I have not used them
#But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass

try:
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    print("\nWARNING: Sklearn OneHotEncoder not installed. You will be unable to use XGBoost for Classification")
    pass



##################### DECODER FUNCTIONS ##########################



##################### Kalman FILTER with Point process observation ##########################

class KalmanFilterPP(object):

    """
    Class for the Kalman Filter Decoder with Point Process Observation
    https://direct.mit.edu/neco/article/16/5/971/6831/Dynamic-Analysis-of-Neural-Encoding-by-Point

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self,Distribution="Poisson"):
        # Poisson is one of the most used model for point process
        self.Distribution=Distribution
        

    def fit(self,X_train,y_train):

        """
        Train Kalman Filter Point Process Decoder

        Parameters
        ----------
        X_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_train.T)

        #number of time bins
        nt=X.shape[1]
        num_neuron = X_train.shape[1]
        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        F=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-F*X1)*(X2-F*X1).T/(nt-1) #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t)
        # use Generalized Linear Model
        # Ref: 
        # In our case, this is the transformation from kinematics to spikes

        y_train_glm = np.concatenate([np.ones([y_train.shape[0],1]), y_train],axis=1)
        H = np.zeros([num_neuron,y_train_glm.shape[1]])
        if self.Distribution == "Poisson":
            ### This is super-easy if we rely on built-in GLM fitting code
            # endog: spike
            for i in range(num_neuron):
                glm_poisson_exp = sm.GLM(endog=X_train[:,i], exog=y_train_glm,
                                        family=sm.families.Poisson())
                pGLM_results = glm_poisson_exp.fit(max_iter=100, tol=1e-6, tol_criterion='params')
                # params[0]: pGLM_const
                # params[1:]: pGLM_stimu
                H[i,:] = pGLM_results.params

        else:
            raise Exception('wrong distribution')

        params=[F,W,H]
        self.model=params

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        F,W,H=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))

        #Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            #Do first part of state update - based on transition matrix
            P_m=A*P*A.T+W
            state_m=A*state

            #Do second part of state update - based on measurement matrix
            K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state) #Record state at the timestep
        y_test_predicted=states.T
        return y_test_predicted
