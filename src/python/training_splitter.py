from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def evaluate(classes, pred):
    accuracy = accuracy_score(y_test, pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    labels = np.array(classes)
    conf_matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# 1. Caricamento del dataset dal file CSV
df = pd.read_csv('kicksplit_dataset_1sample.csv')

class_counts = df['Class'].value_counts()
min_size = class_counts.min()

df_balanced = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(min_size, random_state=42))
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
x = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

# Dataset splitting
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size=0.8,  
                                                    random_state=42)


# Preprocessing
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

class_weights = {
    "Still" : 30,
    "Kick" : 20
}


# Model training
rfc = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weights)
svc = SVC(kernel='rbf', probability=True, class_weight=class_weights)
#gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=42)
#knn = KNeighborsClassifier(n_neighbors=10)

rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
evaluate(["Still", "Kick"], rfc_pred)

svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
evaluate(["Still", "Kick"], svc_pred)

#gbc.fit(x_train, y_train)
#mlp.fit(x_train, y_train)
#knn.fit(x_train, y_train)

'''
estimators = [('svc', svc), ('rfc', rfc)]#, ('gbc', gbc), ('mlp', mlp), ('knn', knn)] 

eclf1 = VotingClassifier(estimators=estimators, voting='soft')
eclf1 = eclf1.fit(x_train, y_train)

ensemble_pred = eclf1.predict(x_test)

evaluate(["Inizio", "Calcio", "Fine"], ensemble_pred)


# Confusion matrix



dump(classifier, 'C:\\Users\\simon\\OneDrive\\Desktop\\Taekwondo-training-system\\Codice\\modello\modello.sav')'''