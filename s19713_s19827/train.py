import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.classification import *

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the training csv file from the URL
    csv_url = ("data\\training_dataset.csv")
    
    try:
        dataset = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Something went wrong during reading the training dataset. Error: %s", e
        ) 
    dataset.columns
    columns = ["Diabetes_012","HighBP","HighChol","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","MentHlth","Sex","Age","Education"]
    data = dataset[columns].copy()
    
    print(dataset.head())
    
    data = dataset.sample(frac=0.8, random_state=800)
    data.reset_index(drop=True, inplace=True)
    
    data_unseen = dataset.drop(data.index)
    data_unseen.reset_index(drop=True, inplace=True)
    
    print('Data to model: ' + str(data.shape))
    print('Data to predict: ' + str(data_unseen.shape))
    
    exp_reg101 = setup(data=data, target='Diabetes_012', session_id=123)
    print(exp_reg101)
    
    best = compare_models()
    results = pull()
    print(results)
    
    lightgbm = create_model('lightgbm')
    modelSummary = pull()
    print(modelSummary)

    tuned_lightgbm = tune_model(lightgbm)
    lightbgmSummary = pull()
    
    print('lightgbm')
    print(lightbgmSummary)
    
    results = evaluate_model(tuned_lightgbm)
    print(results)

    final_lightgbm = finalize_model(tuned_lightgbm)

    predictions = predict_model(final_lightgbm, data=data_unseen)

    print(predictions.head())

    
    pickle.dump(final_lightgbm,open("model/model","wb"))