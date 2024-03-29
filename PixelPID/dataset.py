import numpy as np
import torch
import time
import math
import os
import json
from larcv.config_builder import ConfigBuilder
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from larcv.queueloader import queue_interface
import config

class NumberProtonDataset(Dataset):
    """
    An instance of a Pytorch dataclass which will return images of an interaciton from the three planes of TPC along with 
    the number of protons in interaciton.
    The structure of the labels is one hot encoded where 0 proton -> [1, 0, 0, 0], 1 proton -> [0, 1, 0, 0],
    2 proton -> [0, 0, 1, 0], more than 2 proton -> [0, 0, 0, 1]
    """
    def __init__(self, rootDirectory, name="default", maxVoxels=2000, maxParticles=20, length=1, imageDims=(1280, 2048), nClasses=4, wireAngle=35.7, threeDimentional=False):
        """
        Initialise the larcv batch loader
        """
        
        # Define numbr of datapoints desired
        self.length = length

        # Define parameters of dataset
        self.imageDims = imageDims
        self.nClasses = nClasses
        self.maxVoxels = maxVoxels
        self.maxParticles = maxParticles
        self.batchSize = 1                                  # Batch size will equal 1 as we only want to return one data point       
        self.threeDimentional = threeDimentional
        self.wireAngles = {
            "U":            wireAngle,
            "V":            -wireAngle,
            "W":            0.0
        }

        # Define name of config builder
        self.configBuilderName = name

        # Create list of file names in root directory
        fileNames = os.listdir(rootDirectory)
        self.filePaths = [os.path.join(rootDirectory, fileName) for fileName in fileNames]

        # Generate config for datafiles
        self.generateConfigBuilder()

        # Generate queue manager
        self.generatgeQueueInterface()

    def generateConfigBuilder(self,):
        """
        Generate a config object for larcv queue manager using the specified files
        """

        # Initialise config builder class
        self.configBuilder = ConfigBuilder()
        self.configBuilder.set_parameter(self.filePaths, "InputFiles")              # Pass list containing all files
        self.configBuilder.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
        self.configBuilder.set_parameter(5, "ProcessDriver", "Verbosity")
        self.configBuilder.set_parameter(5, "Verbosity")
    
        # Add particle data to config builder
        self.configBuilder.add_batch_filler(
            datatype  = "particle",
            producer  = "protID",
            name      = self.configBuilderName+"Label",
            MaxParticles = self.maxParticles,
        )

        # Add image data to config builder
        self.configBuilder.add_batch_filler(
            datatype  = "sparse3d",
            producer  = "dunevoxels",
            name      = self.configBuilderName+"Data", 
            MaxVoxels = self.maxVoxels, 
            Augment   = False, 
            Channels  = [1,], 
        )

        # Build up data keys to access data
        self.dataKeys = {
            "image":        self.configBuilderName+"Data",
            "label":        self.configBuilderName+"Label",
        }

    def dumpConfigBuilder(self,):
        """
        Dump the contents of the config builder in the json fromat
        """

        print(json.dumps(self.configBuilder.get_config(), indent=2))

    def generatgeQueueInterface(self,):
        """
        Generate queue interface object to be used to geet new datapoints from dataset
        """

        self.queueInterface = queue_interface(random_access_mode="random_blocks", seed=21062000)
        self.queueInterface.no_warnings()

        # Initialise data manager 

        io_config = {
            'filler_name' : self.configBuilderName,
            'filler_cfg'  : self.configBuilder.get_config(),
            'verbosity'   : 5,
            'make_copy'   : False 
        }

        self.queueInterface.prepare_manager(self.configBuilderName, io_config, self.batchSize, self.dataKeys, color=None)

    def fetchNewDatapoints(self,):
        """
        Request next datapoint dictionary from the queue interface
        """

        self.queueInterface.prepare_next(self.configBuilderName)
        dataDictionary = self.queueInterface.fetch_minibatch_data(self.configBuilderName, pop=True, fetch_meta_data=True)
        return dataDictionary
    
    def pointcloudToImage3D(self, pointcloud,):
        """
        Convert a pointcloud to it's corresponding image of sdimentions defined in constructor of the class
        """

        # Create empty image of specified dimentions
        image = np.zeros(self.imageDims)
        
        # Iterate and assing each point in pointcloud to image
        for point in pointcloud[0]:
            x, y, z, pixelValue = point
            # Check of valid points
            if (x >= 0) and (y >= 0) and (z >= 0):
                image[int(z-1)][int(y-1)][int(x-1)] = pixelValue
        
        return image
    
    def pointcloudToWirePlanes(self, pointcloud, nPlanes=3):
        """
        Convert pointcloud to three wire plane projections
        """

        # Empty array to contain wire plane images, structure of list is [U, V, W]
        wirePlaneImages = []

        # Iterate and assing each point in pointcloud to image
        x, y, z, pixelValue = np.split(pointcloud[0], 4, axis=-1)
        
        # Create mask of non zero index
        nonZeroIndex = np.where(pixelValue != -999)

        # Compute, Assign and save images for each plane
        for idx in range(nPlanes):
            # Initialise empty image
            image = np.zeros(self.imageDims)
            
            # Compute projections
            xProjection = math.cos(list(self.wireAngles.values())[idx])*z[nonZeroIndex] + math.sin(list(self.wireAngles.values())[idx])*y[nonZeroIndex]
            yProjection = x[nonZeroIndex]
            valProjection = pixelValue[nonZeroIndex]

            # Assign values to projection
            for idx in range(len(xProjection)):
                image[int(yProjection[idx])][int(xProjection[idx])] = valProjection[idx]

            # Convert to torch tensor
            image = torch.unsqueeze(torch.from_numpy(image), 0)

            # Append image to projection lists
            wirePlaneImages.append(image)

        return wirePlaneImages


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Return a detector readout and labels
        """

        dataDictionary = self.fetchNewDatapoints()
        
        # Obtain pointcloud from the image label of dictionary
        # Squeeze the first axis as we only have one batch size
        pointcloud = np.squeeze(dataDictionary["image3d"], 0)
        planeU, planeV, planeW = self.pointcloudToWirePlanes(pointcloud) 

        # Apply transformation to input wire plane images
        planeU = config.INPUT_TRANSFORMATION(planeU)
        planeV = config.INPUT_TRANSFORMATION(planeV)
        planeW = config.INPUT_TRANSFORMATION(planeW)

        # Obtain the PDG code of the parent neutrino of the interaction
        nProton = dataDictionary["label"]["_pdg"][0][0]
        # Create empty one hot encoded labels
        label = nProton
        
        return planeU, planeV, planeW, label


class NuIDDataset(Dataset):
    """
    An instance of a Pytorch dataclass which will return images of an interaciton from the three planes of TPC along with 
    the number of protons in interaciton.
    The structure of the labels is one hot encoded where 0 proton -> [1, 0, 0, 0], 1 proton -> [0, 1, 0, 0],
    2 proton -> [0, 0, 1, 0], more than 2 proton -> [0, 0, 0, 1]
    """
    def __init__(self, rootDirectory, name="default", maxVoxels=2000, maxParticles=20, length=1, imageDims=(1280, 2048), nClasses=4, wireAngle=35.7, threeDimentional=False):
        """
        Initialise the larcv batch loader
        """
        
        # Define numbr of datapoints desired
        self.length = length

        # Define parameters of dataset
        self.imageDims = imageDims
        self.nClasses = nClasses
        self.maxVoxels = maxVoxels
        self.maxParticles = maxParticles
        self.batchSize = 1                                  # Batch size will equal 1 as we only want to return one data point       
        self.threeDimentional = threeDimentional
        self.wireAngles = {
            "U":            wireAngle,
            "V":            -wireAngle,
            "W":            0.0
        }

        # Define name of config builder
        self.configBuilderName = name

        # Create list of file names in root directory
        fileNames = os.listdir(rootDirectory)
        self.filePaths = [os.path.join(rootDirectory, fileName) for fileName in fileNames]

        # Generate config for datafiles
        self.generateConfigBuilder()

        # Generate queue manager
        self.generatgeQueueInterface()

    def generateConfigBuilder(self,):
        """
        Generate a config object for larcv queue manager using the specified files
        """

        # Initialise config builder class
        self.configBuilder = ConfigBuilder()
        self.configBuilder.set_parameter(self.filePaths, "InputFiles")              # Pass list containing all files
        self.configBuilder.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
        self.configBuilder.set_parameter(5, "ProcessDriver", "Verbosity")
        self.configBuilder.set_parameter(5, "Verbosity")
    
        # Add particle data to config builder
        self.configBuilder.add_batch_filler(
            datatype  = "particle",
            producer  = "neutID",
            name      = self.configBuilderName+"Label",
            MaxParticles = self.maxParticles,
        )

        # Add image data to config builder
        self.configBuilder.add_batch_filler(
            datatype  = "sparse3d",
            producer  = "dunevoxels",
            name      = self.configBuilderName+"Data3d", 
            MaxVoxels = self.maxVoxels, 
            Augment   = False, 
            Channels  = [1,], 
        )
        # Add image data to config builder
        self.configBuilder.add_batch_filler(
            datatype  = "sparse2d",
            producer  = "dunevoxels",
            name      = self.configBuilderName+"Data2d", 
            MaxVoxels = self.maxVoxels, 
            Augment   = False, 
            Channels  = [0,], 
        )

        # Build up data keys to access data
        self.dataKeys = {
            "image3d":        self.configBuilderName+"Data3d",
            "image2d":        self.configBuilderName+"Data2d",
            "label":        self.configBuilderName+"Label",
        }

    def dumpConfigBuilder(self,):
        """
        Dump the contents of the config builder in the json fromat
        """

        print(json.dumps(self.configBuilder.get_config(), indent=2))

    def generatgeQueueInterface(self,):
        """
        Generate queue interface object to be used to geet new datapoints from dataset
        """

        self.queueInterface = queue_interface(random_access_mode="random_blocks",)
        self.queueInterface.no_warnings()

        # Initialise data manager 

        io_config = {
            'filler_name' : self.configBuilderName,
            'filler_cfg'  : self.configBuilder.get_config(),
            'verbosity'   : 5,
            'make_copy'   : False 
        }

        self.queueInterface.prepare_manager(self.configBuilderName, io_config, self.batchSize, self.dataKeys, color=None)

    def fetchNewDatapoints(self,):
        """
        Request next datapoint dictionary from the queue interface
        """

        self.queueInterface.prepare_next(self.configBuilderName)
        dataDictionary = self.queueInterface.fetch_minibatch_data(self.configBuilderName, pop=True, fetch_meta_data=True)
        return dataDictionary
    
    def pointcloudToImage3D(self, pointcloud,):
        """
        Convert a pointcloud to it's corresponding image of sdimentions defined in constructor of the class
        """

        x, y, z, pixelValue = np.split(pointcloud[0], 4, axis=-1)
        
        # Create mask of non zero index
        nonZeroIndex = np.where(pixelValue != -999)

        # Compute, Assign and save image
        # Initialise empty image
        image = np.zeros(self.imageDims)
            
        # Assign values to projection
        for idx in range(len(x[nonZeroIndex])):
            image[int(z[nonZeroIndex][idx])][int(y[nonZeroIndex][idx])][int(x[nonZeroIndex][idx])] = pixelValue[nonZeroIndex][idx]
        
        return image

    def pointcloudToImage2D(self, pointcloud,):
        """
        Convert a pointcloud to it's corresponding image of sdimentions defined in constructor of the class
        """

        x, y, pixelValue = np.split(pointcloud[0], 3, axis=-1)
        
        # Crude normalisation of pixelValues
        pixelValue /= pixelValue.max()
        
        # Create mask of non zero index
        nonZeroIndex = np.where(pixelValue != -999)

        # Compute, Assign and save image
        # Initialise empty image
        image = np.zeros(self.imageDims)
            
        # Assign values to projection
        for idx in range(len(x[nonZeroIndex])):
            image[int(y[nonZeroIndex][idx])][int(x[nonZeroIndex][idx])] = pixelValue[nonZeroIndex][idx]
        
        return image
    
    def pointcloudToWirePlanes(self, pointcloud, nPlanes=3):
        """
        Convert pointcloud to three wire plane projections
        """

        # Empty array to contain wire plane images, structure of list is [U, V, W]
        wirePlaneImages = []

        # Iterate and assing each point in pointcloud to image
        x, y, z, pixelValue = np.split(pointcloud[0], 4, axis=-1)
        
        # Create mask of non zero index
        nonZeroIndex = np.where(pixelValue != -999)

        # Compute, Assign and save images for each plane
        for idx in range(nPlanes):
            # Initialise empty image
            image = np.zeros(self.imageDims)
            
            # Compute projections
            xProjection = math.cos(list(self.wireAngles.values())[idx])*z[nonZeroIndex] + math.sin(list(self.wireAngles.values())[idx])*y[nonZeroIndex]
            yProjection = x[nonZeroIndex]
            valProjection = pixelValue[nonZeroIndex]

            # Assign values to projection
            for idx in range(len(xProjection)):
                image[int(yProjection[idx])][int(xProjection[idx])] = valProjection[idx]

            # Convert to torch tensor
            image = torch.unsqueeze(torch.from_numpy(image), 0)

            # Append image to projection lists
            wirePlaneImages.append(image)

        return wirePlaneImages


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Return a detector readout and labels
        """

        dataDictionary = self.fetchNewDatapoints()
        
        # Obtain pointcloud from the image label of dictionary
        # Squeeze the first axis as we only have one batch size
        pointcloud = np.squeeze(dataDictionary["image2d"], 0)
        image = self.pointcloudToImage2D(pointcloud)
        image = torch.unsqueeze(torch.from_numpy(image), 0)

        # Apply image pre-processing
        image = config.INPUT_TRANSFORMATION(image)

        # Obtain the PDG code of the parent neutrino of the interaction
        nProton = dataDictionary["label"]["_pdg"][0][0]
        # Create empty one hot encoded labels
        label = nProton
        
        return image, label


def testProton():

    dataset = NumberProtonDataset("/home/giorgio/Desktop/train", name="default", length=1000)
    for idx in range(8):
        fig, axes = plt.subplots(1, 3)
        t1 = time.time()
        U, V, W, label = dataset[idx]
        print(f"Time to generate datapoint: {(time.time()-t1):.3f} seconds")
        print(f"Labels: {label}")
        axes[0].imshow(torch.squeeze(U, 0).numpy(), cmap="jet")
        axes[1].imshow(torch.squeeze(V, 0).numpy(), cmap="jet")
        axes[2].imshow(torch.squeeze(W, 0).numpy(), cmap="jet")
        fig.tight_layout()
        plt.show()

def testNuID():

    dataset = NuIDDataset("/home/giorgio/Desktop/train", name="default", length=1000)
    for idx in range(8):
        t1 = time.time()
        image, label = dataset[idx]
        print(image.shape)
        print(f"Time to generate datapoint: {(time.time()-t1):.3f} seconds")
        print(f"Labels: {label}")
        plt.imshow(torch.squeeze(image, 0).numpy(), cmap="jet")
        plt.show()

if __name__ == "__main__":
    testNuID()