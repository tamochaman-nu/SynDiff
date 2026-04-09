import torch.utils.data
import numpy as np, h5py
import random
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class UnpairedImageDataset(Dataset):
    def __init__(self, root, phase, image_size=256):
        self.dir_A = os.path.join(root, phase + 'A')
        self.dir_B = os.path.join(root, phase + 'B')
        
        # Check if directories exist, if not, try alternative naming
        if not os.path.exists(self.dir_A):
            self.dir_A = os.path.join(root, 'trainA') if phase == 'train' else os.path.join(root, 'testA')
        if not os.path.exists(self.dir_B):
            self.dir_B = os.path.join(root, 'trainB') if phase == 'train' else os.path.join(root, 'testB')

        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        path_A = self.A_paths[index % self.A_size]
        # Unpaired: random sample from B
        index_B = random.randint(0, self.B_size - 1)
        path_B = self.B_paths[index_B]
        
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return img_A, img_B

    def __len__(self):
        return max(self.A_size, self.B_size)


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2', image_size=256):
    # Check if we should use ImageDataset (if subdirectories exist)
    dir_A = os.path.join(input_path, phase + 'A')
    if os.path.exists(dir_A) or os.path.exists(os.path.join(input_path, 'trainA')) or os.path.exists(os.path.join(input_path, 'testA')):
        return UnpairedImageDataset(input_path, phase, image_size=image_size)

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1=LoadDataSet(target_file)
    
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2=LoadDataSet(target_file)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
    return dataset 



#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,3,2))
    data=data.astype(np.float32) 
    if padding:
        pad_x=int((256-data.shape[2])/2)
        pad_y=int((256-data.shape[3])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data
