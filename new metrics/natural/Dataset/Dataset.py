import numpy as np
import cv2
from tensorflow.keras.utils import Sequence


class Dataset:
	def __init__(self, datasetName):
		self.datasetName=datasetName

	def dataloader__daan__gtmodel__conceptdetector(self,batchsize,batch,concept_id):
		return self.dataloader_concept_train(batchsize, batch, concept_id)

class Dataloader__daan__classtest(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_class_test()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_class_test(False, self.batch_size, index)


class Dataloader__daan__explanation3D(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_class_test()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_class_test(True,self.batch_size,index,return3DExplanation=True)


class Dataloader__daan__explanation2D(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_class_test()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_class_test(True,self.batch_size,index)


class Dataloader__daan__concepts(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_concept_train()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_concept_train(self.batch_size, index)


class Dataloader__daan__concepts_test(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_concept_test()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_concept_test(self.batch_size, index)


class Dataloader__daan__concepts_test_lazy(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset
        self.batch_size = batch_size
        self.numexamples = self.dataset.getLength__dataloader_concept_test()
 
    def __len__(self):
        return int(np.ceil(self.numexamples / self.batch_size))

    def __getitem__(self, index):
        return self.dataset.dataloader_concept_test_lazy(self.batch_size, index)

    def __getitem__actual(self, index):
        return self.dataset.dataloader_concept_test(self.batch_size, index)
