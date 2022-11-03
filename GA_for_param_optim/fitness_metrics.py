from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import torch

def accuracy(y_true, y_pred):
	"""
	    Computes the accuracy.
	    Parameters
	    ----------
	    y_true : np.array
	        array of right labels
	    
	    y_pred : np.array
	        array of class confidences for each instance
	    
	    Returns
	    -------
	    accuracy : float
	    	accuracy value
    """


	y_pred_labels = np.argmax(y_pred, axis=1)

	return accuracy_score(y_true, y_pred_labels)



def mse(y_true, y_pred):
	"""
	    Computes the mean squared error (MSE).
	    Parameters
	    ----------
	    y_true : np.array
	        array of right outputs
	    
	    y_pred : np.array
	        array of predicted outputs
	    
	    Returns
	    -------
	    mse : float
	    	mean squared errr
    """

	return mean_squred_error(y_true, y_pred)

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n