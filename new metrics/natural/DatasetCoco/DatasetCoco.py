import numpy as np
import cv2
import os
import glob
import collections
import itertools
import random
import sys
from Dataset.Dataset import Dataset
import tensorflow as tf

class DatasetCoco(Dataset):

	def __init__(self, calc_concepts_in_images=False,folders=["train2017","val2017"]):
		
		super(DatasetCoco, self).__init__("DatasetCoco")

		self.settings= {}
		self.settings['calc_concepts_in_images'] = calc_concepts_in_images
		self.folders = folders
		self.concepts_in_images = {}
		self.concepts = {}
		self.paths_class = {}
		self.labels_class = {}
		self.paths_concept = {}
		self.image_files = {}
		folder_path_segmented  = {}
		folder_path_input_images  = {}



		for folder in self.folders:
			folder_path_segmented[folder] = "/colab_daan_coco/MyDrive/Colab_daan_coco/DatasetCoco/stuffthingmaps_trainval2017/"+folder   # Folder containing segmentation images
			folder_path_input_images[folder] = "/colab_daan_coco/MyDrive/Colab_daan_coco/DatasetCoco/"+folder                            # Folder containing input images

			### Determine the paths of all images ###

			self.image_files[folder] = glob.glob(os.path.join(folder_path_segmented[folder], "*.png"))


			### Determine which concepts are contained in the images ###

			if self.settings['calc_concepts_in_images']:

				concepts_in_images = []

				for image_file in self.image_files[folder]:
					image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
					concepts_in_images.append(self.find_unique_numbers(image))

					if len(concepts_in_images) % 100 == 0:
						print((len(concepts_in_images)/16511)*100,"%")
			 
				with open('/colab_daan_coco/MyDrive/Colab_daan_coco/DatasetCoco/concepts_in_images_'+folder+'.npy', 'wb') as f:
					concepts_in_images = np.array(concepts_in_images, dtype="object")
					np.save(f, concepts_in_images)

			with open('/colab_daan_coco/MyDrive/Colab_daan_coco/DatasetCoco/concepts_in_images_'+folder+'.npy', 'rb') as f:
				concepts_in_images = np.load(f, allow_pickle=True)
				self.concepts_in_images[folder] = list(concepts_in_images)


			### Generate example pairs of image paths and a vector denoting which concepts are in the image ###

			examples = self.genConceptExamples(folder)
			self.paths_concept[folder] = []
			for example in examples:
				self.paths_concept[folder].append(example)




	def resize_image(self, image):  # Preprocess and resize images to make it in the format expected by the model. 
    
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = tf.keras.applications.resnet.preprocess_input(image)

		scale = np.min([image.shape[0], image.shape[1]])
		new_shape = [int(np.ceil(scale * x)) for x in [image.shape[0], image.shape[1]]]
		image = image[0:scale,0:scale]
		image = cv2.resize(image, (224,224))

		return image


	def find_unique_numbers(self, matrix): # Find unique concepts in an image
		unique_numbers = []
		for number in matrix.flatten():
			if number not in unique_numbers and number < 255 and number > 0:
				unique_numbers.append(number)
		return unique_numbers


	def genConceptExample(self, path, folder): # Load concept input image
		return cv2.imread("/colab_daan_coco/MyDrive/Colab_daan_coco/DatasetCoco/"+folder+"/"+path.split("/")[-1].split(".")[0]+".jpg") 


	def genConceptExamples(self,folder): # Generate example pairs of image paths and a vector denoting which concepts are in the image
		result = []
		for path, conceptsInImage in zip(self.image_files[folder], self.concepts_in_images[folder]):
			vector = np.zeros(182)
			for conceptid in range(182):
				if conceptid in conceptsInImage:
					vector[conceptid] = 1
			result.append([path, vector])

		return result



	def dataloader_class_train(self,returnExplanations,batchsize,batch,return3DExplanation=False):
		return None



	def dataloader_class_test(self,returnExplanations,batchsize,batch,return3DExplanation=False):
		return None
		

  
	def dataloader_concept_train(self, batchsize, batch):

		x = []
		y = []
		
		for index, path in enumerate(self.paths_concept["train2017"][batchsize*batch:batchsize*(batch+1)]):
			tmp = self.genConceptExample(path[0], "train2017")
			x.append(self.resize_image(tmp))
			y.append(path[1])

		return 	np.array(x), np.array(y)



	def dataloader_concept_test(self, batchsize, batch):

		x = []
		y = []
		
		for index, path in enumerate(self.paths_concept["val2017"][batchsize*batch:batchsize*(batch+1)]):
			tmp = self.genConceptExample(path[0], "val2017")
			tmp2 = self.resize_image(tmp)
			x.append(tmp2)
			y.append(path[1])
    
		return_x, return_y = np.array(x), np.array(y)

		return 	return_x, return_y



	def dataloader_concept_test_lazy(self, batchsize, batch):

		x = []
		y = []

		for index, path in enumerate(self.paths_concept["val2017"][batchsize*batch:batchsize*(batch+1)]):
			x.append(None)
			y.append(path[1])
    
		return_x, return_y = np.array(x), np.array(y)

		return 	return_x, return_y


	def getLength__dataloader_class_test(self):
		return len(self.labels_class["val2017"])

	def getLength__dataloader_class_train(self):
		return len(self.labels_class["train2017"])

	def getLength__dataloader_concept_train(self):
		return len(self.paths_concept["train2017"])

	def getLength__dataloader_concept_test(self):
		return len(self.paths_concept["val2017"])

	def getLength__dataloader_concept_test_lazy(self):
		return len(self.paths_concept["val2017"])