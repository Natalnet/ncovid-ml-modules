
# Ncovid Functionalites

The Ncovid library is designated for multiple goals. Some of them are listed below 
with their respective basic (in high-level) commands and jupyter-notebook files.

## Usage/Examples

- **For Data Manipulation**

    [Loading local data](loading_local_data.ipynb)

        set configuration
        create data
        collect local data

    [Loading remote data](loading_remote_data.ipynb)

        set configuration
        create data
        collect remote data
    
    [Building train dataset](building_train_dataset.ipynb)

        set the configuration
        load data
        build train

    [Building test dataset](building_test_dataset.ipynb)

        set configuration
        load data
        build test

    [Building train and test dataset](building_train_and_test_dataset.ipynb)

        set configuration
        load data
        build train test

- **For Model Manipulation**

    [Creating model](creating_a_model.ipynb)

        set configuration
        build data
        create model
    
    [Training saving model](training_and_saving_a_model.ipynb)  

        set configurations
        build data
        create model
        fit model
        save model

    [Loading local model](loading_a_local_model.ipynb) 
        
        set configuration
        create model
        load local model
    
    [Loading remote model](loading_a_remote_model.ipynb) 
        
        set configuration
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
