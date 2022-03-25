# Module with the fuctions needed for AIDA ANNs
# Updated to work with tensoflow 1.12 and tf.keras

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
def scale_data(X, Y):
    """ 
        scales data and computes scaler
    """
    poly = PolynomialFeatures(2)
    X_scaler = StandardScaler()
    Y_scaler = MinMaxScaler((0.2, 0.8))
    
    
    X_poly = poly.fit_transform(X) #we now have a feature vector with 15 rows instead of only 4
    X_scaled = X_scaler.fit_transform(X_poly)
    
    Y_scaled = Y_scaler.fit_transform(Y)
    
    scaler_cache = {"X_poly": poly, "X_scaler": X_scaler, "Y_scaler": Y_scaler}
    
    return (X_scaled, Y_scaled, scaler_cache)

def unscale_Y(Y_scaled, scache):
    """ 
    Scales outputs to original values 
    """
    Y = scache["Y_scaler"].inverse_transform(Y_scaled)
    
    return Y

def scale_X(X, scache):
    """ Scales Inputs the same way as the training data """
    
    X_poly = scache["X_poly"].transform(X) #we now have a feature vector with 15 rows instead of only 4
    X_scaled = scache["X_scaler"].transform(X_poly)
    
    return X_scaled

def augment_data(X, Y, target, repetition_factor = 10, scale=0.2):
    """
    Augments data with performance figures that are also meet by the same sizing.
    Augmentation is done by adding point wiht performance that is a rand*scale*mean(Y)
    
    Arguments:
    X -- Circuit target specifications (ANN inputs)
    Y -- Circuit desing variables (ANN outputs)
    target -- array {-1,1}^n_y that indicates if solutions are feasible bellow the original 
              specificaton (-1) or above the original specificaton (1)  
    repetition_factor -- number of repeated points
    scale -- size of the variation
    """
    
    # CAN repeat Y after scaling as they are not modified in the augmentation procedure

    Y_rep = np.repeat(Y, repetition_factor, axis=0)
    X_rep = np.repeat(X, repetition_factor, axis=0)

    m, n_x = X_rep.shape

    # -1 means that specifications with a smaller value are also meet by the design, e.g GDC
    #  1 means that specifications with a larger value are also meet by the design, e.g IDD
    target_scale = scale*np.mean(X, axis = 0)*target


    Y_rep = np.concatenate((Y, Y_rep), axis = 0)
    X_rep = np.concatenate((X, X_rep + np.random.rand(m,n_x)*target_scale), axis = 0)

    return (X_rep, Y_rep)


def augment_specs(X, target, samples = 10, scale=0.2):
    """
    Augments data with performance figures that are also meet by the same sizing.
    Augmentation is done by adding point wiht performance that is a rand*scale*mean(Y)
    
    Arguments:
    X -- Circuit target specifications (ANN inputs)
    Y -- Circuit desing variables (ANN outputs)
    target -- array {-1,1}^n_y that indicates if solutions are feasible bellow the original 
              specificaton (-1) or above the original specificaton (1)  
    repetition_factor -- number of repeated points
    scale -- size of the variation
    """
    
    # CAN repeat Y after scaling as they are not modified in the augmentation procedure

    X_rep = np.repeat(X, samples, axis=0)

    m, n_x = X_rep.shape

    # -1 means that specifications with a smaller value are also acceptable, e.g IDD
    #  1 means that specifications with a larger value are also acceptable, e.g GBW
    #target_scale = scale*X*target
    target_scale = np.repeat(scale*X*target, samples, axis=0)

    X_rep = np.concatenate((X, X_rep + np.random.rand(m,n_x)*target_scale), axis = 0)

    return X_rep

#------------------------------------
# Model definitons
#------------------------------------

def build_dense_model( dims, activation = 'sigmoid', l2_lambda = 0, loss='mean_squared_error'):
    """ 
    Builds the dense model and compiles it.
        
    Arguments:
    layer -- tuple with the number of nodes at each hidden layer. First number is the input size.
    activation = 'sigmoid' -- the activation to be used
    l2_lambda = 0 -- l2 with regularization lambda
    loss='mean_squared_error' -- loss to be used
    
    Returns:
    model -- the keras model
    
    """
    # SizingNet1-Single circuit model
    
    model = tf.keras.models.Sequential()
    level = 0 
    n_prev = dims[0]    
    for n_d in dims[1:]:
        model.add(tf.keras.layers.Dense(n_d, 
                        input_dim=n_prev, 
                        kernel_initializer='normal', 
                        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda), 
                        activation=activation,
                        name='dense'+str(level) + '_' + str(n_d)))
        n_prev = n_d
        level = level + 1
    # Compile model
    model.compile(loss=loss,optimizer='adam',metrics=['mae', 'mse']) #recommended loss and optimizer 
                                                                        #functions for regression
    return model


#deprecated
def baseline_model_reg():
    # SizingNet1-Single circuit model
    
    model = Sequential()
    model.add(Dense(60, input_dim=15, kernel_initializer='normal', kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='sigmoid'))
    model.add(Dense(120, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.00001),  activation='sigmoid'))
    model.add(Dense(12, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.00001), activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse']) #recommended loss and optimizer 
                                                                        #functions for regression
    return model


def predict_circuit(X, model, scalers, scale_inputs=False):
    """ 
    Predict the circuit desing variables for given target specifications.
    
    Args:
    X -- a numpy 2D array with one target specification per row 
    model -- the trained model
    scalers -- the cache of scaller usef for pre-processing
    
    
    Returns the predicted desing variables.
    """
    
    if scale_inputs :
        X = scale_X(X, scalers)
        
    Y_pred = unscale_Y(model.predict(X), scalers)
    return Y_pred



# utils

def format_circuit(y, labels, minVal=None, maxVal=None):
    """ 
    Formats the desing variables in an aida friendly string. 
    
    Arguments:
    y -- a 2D numpy array with one solution per row
    lables -- a list of string with variables names in aida setup
    
    Returns -- dseing variables as a Stirng that AIDA can parse
    """
    s = ""
    n_r, n_c = y.shape
    if minVal is not None:
        y = np.clip(y, minVal, maxVal)

        
    for r in range(n_r):
        s += "Design Variables\n----\n"
        for l,c in zip(labels, range(n_c)):
            s += l + " = " + str(y[r][c]) + "\n"
        s+="----\n"
    return s



# Grid Search Setup 
def grid_search (dims = [(15, 60, 120, 12),(15, 30, 60, 12),(15, 120, 240,120,12)],
                 activations = ['sigmoid', 'tanh', 'relu'],
                 reg_lambdas = [0.001, 0.0003, 0.0001, 0.00003, 0.00001],
                 ):
    grid = []

    for activation in activations:
        for reg_lambda in reg_lambdas:
            for dim in dims:
                model = build_model(dim, activation=activation, reg_l2_lambda=reg_lambda)

                #Starting Time
                start_time = datetime.now()
                print('Start ' + str(dim) + ' ' + str(activation) + ' ' + str(reg_lambda) +  ':')

                history = model.fit(X_train, 
                    y_train, 
                    validation_data = (X_test,y_test),
                    epochs = 500, 
                    batch_size= 512, 
                    verbose = 0)
            
                print('loss-train\t',history.history['loss'][-1])
                print('loss-val \t', history.history['val_loss'][-1])
                stop_time = datetime.now()
                elapsed_time = stop_time - start_time
                print('Elapsed Time:', elapsed_time) 

                grid.append( (model, history, dim, activation, reg_l2_lambda))
     
    return grid



def show_history(history, metrics=['mae'], save_fig=False):
    """
    Prints the history
    
    Arguments:
    history -- the history as returned by model.fit
    """
    print('loss-train\t',history.history['loss'][-1])
    print('loss-val \t', history.history['val_loss'][-1])
    
    # summarize history for loss
    plt.plot((history.history['loss']))
    plt.plot((history.history['val_loss']))
    plt.title('model loss')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save_fig:
        plt.savefig('loss.png', dpi=300)
    plt.show()
        
    for metric in metrics:
        print(metric + '-train\t', history.history[metric][-1])
        print(metric + '-val \t', history.history['val_'+metric][-1])
        
        # summarize history for metric
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.title('model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save_fig:
            plt.savefig(metric + '.png', dpi=300)
        plt.show()

