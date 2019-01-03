# deeplearningChexReproduce

To retrain the model first run train_global.py, then train_local.py and lastly train_concat.py. You can also retrain the model in the Reproduce.ipynb notebook. Please ensure that you are running the scripts in the correct directory. Be aware that running these scripts takes a very long time. 

For retraing the full uncompressed images must be downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC. In the notebook you will need to put in the path to the images. It is shown where to write the path.

To run on windows num_workers must be set to 0 in the dataloaders.

The model state files are 27MB, and there's a limit to 25MB - write to us at s136587@student.dtu.dk or s144137@student.dtu.dk and we can forward the state files, producing or results for the global and local branches. You will need these files for the notebook.

To the teachers of this course the state files are uploaded on Inside along with the report. Download the state files and put them in the same directory as the files from this github repo.
