# deeplearningChexReproduce

To retrain the model first run train_global.py, then train_local.py and lastly train_concat.py. Please ensure that you are running the scripts in the correct directory. Be aware that running these scripts takes a very long time. 

For retraing the full uncompressed images must be downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC and paths needs to be updated in the training, model, evaluating and dataset files in order to successfully run the code and save the states. 

To run on windows num_workers must be set to 0 in the dataloaders in the dataloaders.

The model checkpoints files are 27MB, and we theeres a limit to 25MB - write to us at s136587@student.dtu.dk or s144137@student.dtu.dk and we can forward the checkpoints, producing or results for the global and local branches. The checkpoint for the fusion branch, is the one called "checkpoint_t0.5_m=False_fusion"
