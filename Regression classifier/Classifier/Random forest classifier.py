import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from support_functions import plot_confusion_matrix, generate_features_targets
from sklearn.ensemble import RandomForestClassifier

data = np.load('E:\COURSES\Data driven astronomy-Coursera\Regression classifier\Classifier\galaxy_catalogue.npy')

def splitdata_train_test(data, fraction_training):
    np.random.seed(0)
    np.random.shuffle(data)
    split = int(data.shape[0]*fraction_training)
    training_set = data[:split]
    testing_set = data[split:]
    return (training_set,testing_set)

def dtc_predict_actual(data):
  training_set, testing_set = splitdata_train_test(data,0.7)
  train_features, train_targets = generate_features_targets(training_set)
  test_features, test_targets = generate_features_targets(testing_set)
  dtr = DecisionTreeClassifier()
  dtr.fit(train_features, train_targets)
  predictions = dtr.predict(test_features)
  return (predictions,test_targets)

def calculate_accuracy(predicted, actual):
  bc = sum(t == p for t, p in zip(actual, predicted))  
  ac = bc/len(actual)
  return ac
  pass


dtc = DecisionTreeClassifier()
features, targets = generate_features_targets(data)
predicted = cross_val_predict(dtc, features, targets, cv=10)
model_score = calculate_accuracy(predicted, targets)
class_labels = list(set(targets))
model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
plt.show()

n_estimators = 50              # Number of trees
rfc = RandomForestClassifier(n_estimators=n_estimators)

def rf_predict_actual(data, n_estimators):
  features, targets = generate_features_targets(data)
  rfc = RandomForestClassifier(n_estimators=n_estimators)
  predicted = cross_val_predict(rfc, features, targets, cv=10)
  return (predicted, targets) 

predicted, actual = rf_predict_actual(data, n_estimators)
accuracy = calculate_accuracy(predicted, actual)
class_labels = list(set(actual))
model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
plt.show()

