import cxr_dataset as CXR
import eval_model as E
import model_local as ML
import sys


method = 'Complex'
thresh = 0.5
mean = False

heatmap_methods = [method, thresh,mean]

print('Running local branch with following parameters', heatmap_methods)

base_path = 'results_local/'


state_dict_path = base_path + 'state_dict_t' + str(thresh) + '_m=' + str(mean)
model_path = base_path + 'checkpoint_t' + str(thresh) + '_m=' + str(mean)
aucs_path = base_path + 'aucs_t' + str(thresh) + '_m=' + str(mean) + '.csv'
preds_path = base_path + 'preds_t' + str(thresh) + '_m=' + str(mean) + '.csv'

paths = [state_dict_path, model_path, aucs_path, preds_path]




# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
#PATH_TO_IMAGES = r'/enter_your_path_to_downloaded_uncompressed_nih_images_here/"
PATH_TO_IMAGES = "/work3/s144137/DL/extracted_images/images"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
state_dict = 'results_global/state_dict'


preds, aucs, model = ML.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY,state_dict,heatmap_methods, paths)

