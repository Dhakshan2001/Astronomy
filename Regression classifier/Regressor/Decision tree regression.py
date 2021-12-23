import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

data = np.load('sdss_galaxy_colors.npy')
dtr = DecisionTreeRegressor(max_depth=19)

def get_features_targets(data):
    '''Classifies and gives out the raw input data by filters and its redshifts'''
    features = np.zeros((data.shape[0], 4))
    features[:, 0] = data['u'] - data['g']
    features[:, 1] = data['g'] - data['r']
    features[:, 2] = data['r'] - data['i']
    features[:, 3] = data['i'] - data['z']
    targets = data['redshift']
    return (features,targets)

features, targets = get_features_targets(data)

def median_diff(predicted, actual):
    '''Calculates the differenence in median between predicted and actual values'''
    med_diff = np.median(abs(predicted-actual))
    return med_diff
    pass

def accuracy_by_treedepth(features, targets, depths):
    '''Used to calculate the accuracy of the regressor by comparing the median differences
    between training and test data'''
    # split the data into testing and training sets
    split1 = features.shape[0]//2
    train_features = features[:split1]
    test_features = features[split1:]
    split2 = targets.shape[0]//2
    train_targets = targets[:split2]
    test_targets = targets[split2:]
    
    # initialise arrays or lists to store the accuracies for the below loop
    med_diff_train=[]
    med_diff_test=[]
    
    # loop through depths
    for depth in depths:
        # initialize model with the maximum depth. 
        dtr = DecisionTreeRegressor(max_depth=depth)

        # train the model using the training set
        dtr.fit(train_features, train_targets)
        # get the predictions for the training set and calculate their median_diff
        prediction1 = dtr.predict(train_features)
        m_diff1 = median_diff(train_targets, prediction1)
        med_diff_train.append(m_diff1)
        # get the predictions for the testing set and calculate their median_diff
        prediction2 = dtr.predict(test_features)
        m_diff2 = median_diff(test_targets, prediction2)
        med_diff_test.append(m_diff2)
    # return the accuracies for the training and testing sets
    return med_diff_train,med_diff_test

    
# Generate several depths to test
tree_depths = [i for i in range(1, 36, 2)]

# Call the function
train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
print("Depth with lowest median difference : {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))
  
# Plot the results
train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Median of Differences")
plt.legend()
plt.show()


def cross_validate_model(model, features, targets, k):
  '''Used to check the accurcy of the regressor'''
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  med_diff=[]

  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    dtr.fit(train_features,train_targets)
    # predict using the model
    predictions = dtr.predict(test_features)
    # calculate the median_diff from predicted values and append to results array
    m = median_diff(test_targets, predictions)
    med_diff.append(m)
 
  # return the list with your median difference values
  return med_diff


#Cross validating the results

def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    dtr.fit(train_features, train_targets)
    # predict using the model
    predictions = dtr.predict(test_features)
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions    

predictions = cross_validate_predictions(dtr, features, targets, 10)

#We can use the cross_val_predict function from sklearn.model_selection which can do the 
#same as our cross_validate_predictions functions. It is called in the following way 

def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10))

predictions = cross_val_predict(dtr, features, targets, cv=10)

diffs = median_diff(predictions,targets)

# plot the results to see how well our model looks
plt.scatter(targets, predictions, s=0.4)
plt.xlim((0, targets.max()))
plt.ylim((0, predictions.max()))
plt.xlabel('Measured Redshift')
plt.ylabel('Predicted Redshift')
plt.show()


#Some galaxies appear apparently brughter due to the prescence of an active nucleus
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  galaxies = data[data['spec_class'] == b'GALAXY']
  qso = data[data['spec_class'] == b'QSO']
  return (galaxies,qso)

galaxies, qsos= split_galaxies_qsos(data)
qso_med_diff = cross_validate_median_diff(qsos)
qso_med_diff = cross_validate_median_diff(qsos)






