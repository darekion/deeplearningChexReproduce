# deeplearningChexReproduce

To retrain the model first run train_global.py, then train_local.py and lastly train_concat.py

Paths needs to be updated in the training, model, evaluating and dataset files in order to successfully run the code and save the states. 

To run on windows num_workers must be set to 0 in the dataloaders in the model classes.
