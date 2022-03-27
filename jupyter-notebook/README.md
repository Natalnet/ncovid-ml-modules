
# NCovid Modules

The Ncovid library is designed for several purposes. Some of them are listed below with their respective basic (high-level) commands and Jupyter-notebook files.

## Usage/Examples

- **Data Manipulation**

    [Loading local data](loading_local_data.ipynb)

        set and import project configurations
        collect local data

    [Loading remote data](loading_remote_data.ipynb)

        set and import project configurations
        collect remote data
    
    [Building train dataset](building_train_dataset.ipynb)

        set and import project configurations
        collect local data
        build data train

    [Building test dataset](building_test_dataset.ipynb)

        set and import project configurations
        collect local data
        build data test

    [Building train and test dataset](building_train_and_test_dataset.ipynb)

        set and import project configurations
        collect local data
        build data train and test

- **Model Manipulation**

    [Creating model](creating_a_model.ipynb)

        set and import project configurations
        build data
        create model
    
    [Training saving model](training_and_saving_a_model.ipynb)  

        set and import project configurations
        build data
        create model
        fit model
        save model

    [Loading local model](loading_a_local_model.ipynb) 
        
        set and import project configurations
        create model
        load local model
    
    [Loading remote model](loading_a_remote_model.ipynb) 
        
        set and import project configurations
        create model
        load remote model

    [Testing model](testing_a_model.ipynb)

        build test
        load model
        model predict

    [Extracting metric](extracting_a_metric.ipynb)

        test model
        get metric

- **For Model Prediction**

    [Off-line predicting](off-line_predicting.ipynb)

        create predictor
        model predict

    [On-line predicting](on-line_predicting.ipynb)

        run routes.py
        request
        
- **For Model Evaluation**

    [Evaluating a model](evaluating_model.ipynb)

        build test
        load model
        create evaluator
        model evaluator
    
    [Evaluating N times](evaluating_n_times.ipynb)

        build train
        build test
        create model
        create evaluator
        model evaluator

    [Evaluating N models N times](evaluating_n_models_n_times.ipynb)

        set variations
        set metrics
        create evaluator
        model evaluator
