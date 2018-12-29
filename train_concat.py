import model_concat as M
import sys

method = 'Complex'
thresh = 0.5
mean = False
heatmap_methods = [method, thresh,mean]

local_base_path = 'results_local/state_dict'
global_state_dict = 'results_global/state_dict'
local_state_dict = local_base_path + '_t' + str(thresh) + '_m=' + str(mean)

fusion_base_path = 'results_fusion/'
state_dict_path = fusion_base_path + 'state_dict_t' + str(thresh) + '_m=' + str(mean)
model_path = fusion_base_path + 'checkpoint_t' + str(thresh) + '_m=' + str(mean)
aucs_path = fusion_base_path + 'aucs_t' + str(thresh) + '_m=' + str(mean) + '.csv'
preds_path = fusion_base_path + 'preds_t' + str(thresh) + '_m=' + str(mean) + '.csv'
paths = [state_dict_path, model_path, aucs_path, preds_path]

path_to_images = '/work3/s144137/DL/extracted_images/images' # Set path to images

LR = 0.01
weight_decay = 1e-4

print('Running fusion branch with following parameters: ', heatmap_methods)


model,preds,aucs = M.train_cnn(path_to_images,LR,weight_decay,global_state_dict,local_state_dict,heatmap_methods,paths)
