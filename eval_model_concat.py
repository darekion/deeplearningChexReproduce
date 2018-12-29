import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import cropping as C


def make_pred_multilabel(data_transforms, model, global_model,local_model, PATH_TO_IMAGES, heatmap_methods, paths):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """
    method = heatmap_methods[0]
    thresh = heatmap_methods[1]
    mean = heatmap_methods[2]
    
    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs_global, labels, _ = data
                
        try:
            inputs_local = C.crop_batch(inputs_global,global_model,thresh,method, mean = mean).cuda()
        except:
            print('Cropping failed in training validation loop')
            
        true_labels = labels.cpu().data.numpy()
        
        batch_size = inputs_global.shape[0]
        inputs_global = inputs_global.cuda()
        labels = Variable(labels.cuda()).float()
        
        features = global_model.features(inputs_global)
        global_vector = F.relu(features,inplace=True)
        global_vector = F.avg_pool2d(global_vector,kernel_size=7,stride=1).view(features.size(0),-1)                
        
        features = local_model.features(inputs_local)
        local_vector = F.relu(features)
        local_vector = F.avg_pool2d(local_vector,kernel_size=7,stride=1).view(features.size(0),-1)
        
        concat = torch.cat((global_vector,local_vector),1)
        concat = Variable(concat.cuda())
        
        outputs = model(concat)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
                'Hernia']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv(paths[3], index=False)
    auc_df.to_csv(paths[2], index=False)
    return pred_df, auc_df
