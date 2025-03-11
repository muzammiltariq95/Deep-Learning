import pandas as pd

data = pd.read_csv('bank_note_data.csv')

data.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class',data=data)
plt.show()

sns.pairplot(data,hue='Class')
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()

X = df_feat
y = data['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow as tf
df_feat.columns

image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy = tf.feature_column.numeric_column('Entropy')

feat_cols = [image_var,image_skew,image_curt,entropy]

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)

classifier.train(input_fn=input_func,steps=500)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)

note_predictions = list(classifier.predict(input_fn=pred_fn))

note_predictions[0]

final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import classification_report,confusion_matrix