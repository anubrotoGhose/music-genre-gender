# music-genre-gender

# Music Genre Recommendation System

This project is a simple Machine Learning application that predicts a person's preferred music genre based on their age and gender using a Decision Tree Classifier from Scikit-learn.

---

## Features

- Load and preprocess music dataset
- Train a Decision Tree model
- Evaluate model accuracy
- Save and load trained model using Joblib
- Predict music genre for new users
- Export decision tree visualization

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Google Colab

---

## Dataset

The dataset file used:

```bash
music.csv
````

Example structure:

| age | gender | genre     |
| --- | ------ | --------- |
| 20  | 1      | HipHop    |
| 23  | 1      | Jazz      |
| 25  | 0      | Classical |

### Gender Encoding

* `1` → Male
* `0` → Female

---

## Installation

Install required libraries:

```bash
pip install pandas scikit-learn joblib
```

---

## Training the Model

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')

x = music_data.drop(columns=['genre'])
y = music_data['genre']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)

print(score)
```

---

## Model Accuracy

The model achieved:

```bash
Accuracy: 1.0
```

---

## Saving the Model

```python
import joblib

joblib.dump(model, 'music-recommender.joblib')
```

---

## Loading the Model

```python
import joblib

model = joblib.load('music-recommender.joblib')

predictions = model.predict([[21, 1]])

print(predictions)
```

Example output:

```bash
['HipHop']
```

---

## Exporting Decision Tree Visualization

```python
from sklearn import tree

tree.export_graphviz(
    model,
    out_file='music-recommender.dot',
    feature_names=['age', 'gender'],
    class_names=sorted(y.unique()),
    label='all',
    rounded=True,
    filled=True
)
```

This generates a `.dot` file that can be visualized using Graphviz.

---

## Project Structure

```bash
music-recommender/
│
├── music.csv
├── music-recommender.joblib
├── music-recommender.dot
├── train_model.py
├── predict.py
└── README.md
```

---

## Future Improvements

* Add more features like mood, tempo, or listening history
* Build a web app using Flask or Streamlit
* Use advanced ML algorithms
* Deploy the model online

---

## Author

Developed by Anubroto Ghose using Python and Scikit-learn for learning Machine Learning basics.
