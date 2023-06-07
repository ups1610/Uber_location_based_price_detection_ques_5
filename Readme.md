## End to End ML Project

### created a environment
```
conda create -p venv python==3.8
```

### Install all necessary librries
```
pip insatll -r requirements.txt
```
### Install the entire package
```
python setup.py install
```

## upload the data

The data can be uploaded in the folder in csv format

```
notebooks/data
```

### About the project

Uber is a taxi service provider as we know, we need to predict the high

booking area using an Unsupervised algorithm and price for the location using a

supervised algorithm and use some map function to display the data


### Description

The above dataset get cleaned and preprocessed using label various techniques.

Since from the observation and by doing EDA analysis, its a Unsupervised and supervised machine learning problem so we apply 

K-means and such Random Forest Regressor  on the dataset with an accuracy of 96.23%

The project analysis can be present in a file :

```
EDA.ipynb
model_trainer.ipynb
```

### Project run command

```
cd Uber_Ques_5
python src/pipeline/training_pipeline.py
```

## Model 

Models are found in

```
artifcats folder
```

### Get the map data in form of frontend

The map data present in html file and present in 

```
artifacts/map.html
```
