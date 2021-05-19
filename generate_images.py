import json
import csv
import cv2
from rdkit.Chem import Draw
from rdkit import Chem
from tqdm import tqdm
from utils import *
from pqdm.processes import pqdm

def write_smiles_and_imgs(smiles_dict_train, smiles_dict_val, smiles, idx):
    smiles_dict = None
    if idx >= 1980000:
        mode = 'val'
        idx -= 1980000
        smiles_dict = smiles_dict_val
    else:
        mode = 'train'
        smiles_dict = smiles_dict_train
    smiles_dict[str(idx)] = smiles
    #print(f'smiles_dict_{mode} has {len(smiles_dict.keys())} terms.')
    img_path = base_path + f'/oscar_images/{mode}/{idx}.png'
    if not os.path.exists(img_path):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, img_path)
        originalImage = cv2.imread(img_path)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(img_path, blackAndWhiteImage)
 
 
 
def write_imgs(img_name, smiles, mode):
    img_path = base_path + f'/oscar_images/{mode}/{img_name}'
    if not os.path.exists(img_path):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, img_path)
        originalImage = cv2.imread(img_path)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(img_path, blackAndWhiteImage)

if __name__ == '__main__':
    base_path = '/ocean/projects/tra190016p/ylix/dacon'
    with open("smiles_json_30000.json", 'r') as fr:
        #cr = csv.reader(fr)
        #next(cr)
        #smiles_json_train, smiles_json_val = {}, {}
        smiles_dict = json.load(fr)
        params = [
            [k,
            smiles_dict[k],
            'test'
            ] for k in smiles_dict.keys()]
            
        pqdm(params,
            write_imgs,
            n_jobs=128,
            argument_type='args',
            desc=f'{color.BLUE}Creating SMILES and images for OSCAR.')

