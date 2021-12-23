import numpy as np

data = np.load('sdss_galaxy_colors.npy')

#to split the training data into input features and their corresponding targets


def get_features_targets(data):
    features = np.zeros((data.shape[0], 4))
    features[:, 0] = data['u'] - data['g']
    features[:, 1] = data['g'] - data['r']
    features[:, 2] = data['r'] - data['i']
    features[:, 3] = data['i'] - data['z']
    targets = data['redshift']
    return (features,targets)

features,targets = get_features_targets(data)
#To predict values based on existing ones using Decision tress
    
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor
dtr.fit(features, targets)
predictions = dtr.predict(features)

#To calculate the difference between predicted and actual values
    
def median_diff(predicted, actual):
  med_diff = np.median(abs(predicted-actual))
  return med_diff
  pass

# write a function that splits the data into training and testing subsets
# trains the model and returns the prediction accuracy with median_diff
tree_depths = [3,5,7]  

def validate_model(model, features, targets):
    # split the data into training and testing features and predictions
    split1 = features.shape[0]//2
    train_features = features[:split1]
    test_features = features[split1:]
    split2 = targets.shape[0]//2
    train_targets = targets[:split2]
    test_targets = targets[split2:]
    
    # train the model
    dtr.fit(train_features, train_targets)
    
    # get the predicted_redshifts
    predictions = dtr.predict(test_features)
    
    # use median_diff function to calculate the accuracy
    return median_diff(test_targets, predictions)

#To plot a graph of colour indices to get the redshift

import matplotlib.pyplot as plt

x = data['u'] - data['g']
y = data['r'] - data['i']

redshift = data['redshift']
t = len(redshift)

plt.xlabel("Colour index u-g")
plt.ylabel("Colour index r-i")
plt.title("Redshift (colour) u-g versus r-i")
plt.scatter(x, y, c=t, linewidths=0,cmap="Oranges_r")
plt.colorbar()



