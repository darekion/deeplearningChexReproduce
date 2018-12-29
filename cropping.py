import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage

def resize(im,size):
    im_resize = cv2.resize(im,size,interpolation=cv2.INTER_CUBIC)
    return im_resize


def crop_batch(batch,model,threshold,method,mean):
    batch_size = np.shape(batch)[0]
    inputs = torch.zeros(1,3,224,224)
    for i in range(batch_size):
        image = batch[i,:,:,:].view(1,3,224,224).float().cuda().detach()
        out = model.features(image).detach()
        out = F.relu(out,inplace=True)
        
        if mean:
            heat_map = out.abs().mean(1)[0].view(7,7)
        elif not mean:
            heat_map = out.abs().max(1)[0].view(7,7)
        heat_map = resize(heat_map.cpu().numpy(),(224,224))
        heat_map = cv2.normalize(heat_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        heat_map[heat_map < threshold] = 0
        heat_map[heat_map >= threshold] = 1
        
        
        
        if method == 'Simple':
            #Return without doing further operations
            mask = np.dstack((heat_map,heat_map,heat_map))
            image_masked = mask*image.view(3,224,224).permute(1,2,0).cpu().numpy()
            
            
            image_masked = torch.from_numpy(image_masked).permute(2,0,1).view(1,3,224,224)
            inputs = torch.cat((inputs,image_masked),0)
        elif method == 'Complex':
            labeled_image, num_features = ndimage.label(heat_map)
            sizes = ndimage.sum(heat_map, labeled_image, range(num_features+1))
            largest_blob = sizes.argmax()-1
            objs = ndimage.find_objects(labeled_image,max_label=largest_blob)
            bb = image.view(3,224,224).permute(1,2,0).cpu().numpy()[objs[0]]
            
            dim = np.shape(bb)
            largest_dim = np.argmax(dim[0:2])

            if largest_dim == 0:
                diff = dim[0] - dim[1]
                padding_sides = int(np.round(diff/2))
                padding_top = int(0)
            elif largest_dim == 1:
                diff= dim[1] - dim[0]
                padding_sides = int(0)
                padding_top = int(np.round(diff/2))
            

            color = [0, 0, 0]
            padded_im = cv2.copyMakeBorder(bb, padding_top, padding_top, 
                                           padding_sides, padding_sides, cv2.BORDER_CONSTANT, value=color)
            
            image_cropped = resize(padded_im,(224,224))
            image_cropped = torch.from_numpy(image_cropped).permute(2,0,1).view(1,3,224,224)
            inputs = torch.cat((inputs,image_cropped),0)
            
            
            
  
        
    return inputs[1:,:,:,:]
    
        
        
        
        
        
        
        