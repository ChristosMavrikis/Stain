import argparse
import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import os
import sys
def png_converter(img):

    file = img
    if os.path.exists(file):
        filename=file.split(".")
        img = Image.open(file)
        target_name = filename[0] + ".png"
        rgb_image = img.convert('RGB')
        rgb_image.save(target_name)
        print("Converted image saved as " + target_name)
    # else:
    #     print(file + " not found in given location")
    return target_name

def normalizeStaining(img, step ,saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+str(step)+'.tiff')
        Image.fromarray(H).save(saveFile+str(step)+'_H.tiff')
        Image.fromarray(E).save(saveFile+str(step)+'_E.tiff')

    return Inorm, H, E


def parser_image(path,i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFile', type=str, default=path, help='RGB image file')
    parser.add_argument('--saveFile', type=str, default='output', help='save file')
    parser.add_argument('--Io', type=int, default=240)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    args = parser.parse_args()

    img = np.array(Image.open(args.imageFile))
    print(img.shape)

    normalizeStaining(img=img,
                      step=i,
                      saveFile=args.saveFile,
                      Io=args.Io,
                      alpha=args.alpha,
                      beta=args.beta)
    
if __name__=='__main__':
    i = 0
    for filename in os.listdir('my_imgs'):
        png_img = png_converter('my_imgs/'+filename)
        print(png_img)
        parser_image(png_img,i)
        i=i+1
    #png_converter('imgs/002.tiff')
    # parser_image('imgs/002.png')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--imageFile', type=str, default='imgs\ex03.png', help='RGB image file')
    # parser.add_argument('--saveFile', type=str, default='output', help='save file')
    # parser.add_argument('--Io', type=int, default=240)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--beta', type=float, default=0.15)
    # args = parser.parse_args()
    #
    # img = np.array(Image.open(args.imageFile))
    # print(img.shape)
    #
    # normalizeStaining(img = img,
    #                   saveFile = args.saveFile,
    #                   Io = args.Io,
    #                   alpha = args.alpha,
    #                   beta = args.beta)
