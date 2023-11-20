import numpy as np
import torch
import os
import json
from larcv.config_builder import ConfigBuilder
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from larcv.queueloader import queue_interface

class NumberProtonDataset(Dataset):
    """
    An instance of a Pytorch dataclass which will return images of an interaciton from the three planes of TPC along with 
    the number of protons in interaciton.
    The structure of the labels is one hot encoded where 0 proton -> [1, 0, 0, 0], 1 proton -> [0, 1, 0, 0],
    2 proton -> [0, 0, 1, 0], more than 2 proton -> [0, 0, 0, 1]
    """
    def __init__(self, rootDirectory, maxVoxels=2000, maxParticles=20, length=1, imageDims=1024, nClasses=4):
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

        # Define name of config builder
        self.configBuilderName = "default"

        # Create list of file names in root directory
        fileNames = os.listdir(rootDirectory)
        self.filePaths = [os.path.join(rootDirectory, fileName) for fileName in fileNames]

        # Generate config for datafiles
        self.generateConfigBuilder()

        # Generate queue manager
        self.generatgeQueueInterface()

    def generateConfigBuilder(self, configBuilderName="default"):
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
            datatype  = "sparse2d",
            producer  = "dunevoxels",
            name      = self.configBuilderName+"Data", 
            MaxVoxels = self.maxVoxels, 
            Augment   = False, 
            Channels  = [0,], 
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
    
    def pointcloudToImage(self, pointcloud,):
        """
        Convert a pointcloud to it's corresponding image of sdimentions defined in constructor of the class
        """

        # Create empty image of specified dimentions
        image = np.zeros((self.imageDims, self.imageDims))
        
        # Iterate and assing each point in pointcloud to image
        for point in pointcloud[0]:
            x, y, pixelValue = point
            # Check of valid points
            if (x >= 0) and (y >= 0):
                image[int(y-1)][int(x-1)] = pixelValue
        
        return image


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Return a detector readout and labels
        """

        dataDictionary = self.fetchNewDatapoints()
        
        # Obtain pointcloud from the image label of dictionary
        # Squeeze the first axis as we only have one batch size
        pointcloud = np.squeeze(dataDictionary["image"], 0)
        image = torch.tensor(self.pointcloudToImage(pointcloud))

        # Obtain the PDG code of the parent neutrino of the interaction
        nProton = dataDictionary["label"]["_pdg"][0][0]
        # Create empty one hot encoded labels
        label = torch.zeros(self.nClasses)
        # Fill one hot encoded label according to nuPDG
        match nProton:
            case 0:
                label[0] = 1
            case 1:
                label[1] = 1
            case 2:
                label[2] = 1
            case _:
                label[3] = 1

        return image, label

def test():

    dataset = NumberProtonDataset("/home/giorgio/Desktop/train", imageDims=2000, length=1000)
    label_sum = torch.zeros(4)
    for idx in range(5):
        image, label = dataset[idx]
        label_sum+=label
        print(label)
        plt.imshow(image, cmap="jet")
        plt.show()
    print(label_sum)


if __name__ == "__main__":
    test()