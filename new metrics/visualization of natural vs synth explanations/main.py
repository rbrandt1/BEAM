import cv2
import numpy as np


colormap = cv2.imread("./colormap.png")
colormap = colormap[:,0,:]

def convertPixel(pixel):
    pixel_array = np.array(pixel)

    diff_squared = np.zeros_like(colormap)
    for row in range(colormap.shape[0]):
    	diff_squared[row,:] = colormap[row,:] - pixel_array

    squared_distances = np.sum(diff_squared, axis=1)

    most_similar_row_index = np.argmin(squared_distances)

    return (most_similar_row_index / (colormap.shape[0]-1)) * 2 - 1


def readimage(imgpath):
	img = cv2.imread(imgpath)

	img_result = np.zeros((img.shape[0], img.shape[1]))

	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			img_result[row,col] = convertPixel(img[row,col,:])
	return img_result





metrics = ["GradCAM","GradCAMPP","Saliency","DeconvNet", "GradientInput", "GuidedBackprop", "IntegratedGradients", "SmoothGrad", "SquareGrad","VarGrad","Occlusion","Rise","KernelShap"]

metric_names_row1 = ["Grad-","Grad-","Saliency","Deconv-", "Gradient-", "Guided-", "Integrated-", "Smooth-", "Square-","Var-","Occlu-","Rise","Kernel-"]
metric_names_row2 = ["CAM","CAMPP","","Net", 				"Input", "Backprop", 	"Gradients", "Grad", 	"Grad",	   "Grad","sion","Rise","Shap"]

print("Input & GT & ",end='')

for metric in metric_names_row1:
	print(metric,"&",end='')
print("\\\\")
print(" &  & ",end='')
for metric in metric_names_row2:
	print(metric,"&",end='')
print("\\\\")

for dataset in ["synth","natural"]:
	for cnc in ["C","NC"]:
		if cnc == "C":
			if "synth" in dataset:
				print("\\includegraphics[width=0.05\\textwidth]{real_world_synth_vis/input_synth.png} &",end='')
				print("\\includegraphics[width=0.05\\textwidth]{real_world_synth_vis/gt_synth.png} &",end='')
			else:
				print("\\includegraphics[width=0.05\\textwidth]{real_world_synth_vis/input_natural.png} &",end='')
				print("\\includegraphics[width=0.05\\textwidth]{real_world_synth_vis/gt_natural.png} &",end='')
		else:
			print("&&",end='')

		for metric in metrics:
			print("\\includegraphics[width=0.05\\textwidth]{real_world_synth_vis/"+dataset+"_"+metric+"_"+cnc+".png} &",end='')
		print("\\\\")
	print("\\\\")


class_idx = {"synth":0,"natural":2}


for dataset in ["synth","natural"]:

	for metric in metrics: 

		if "synth" in dataset:
			exp = readimage("./../resnet/paper_images/selected/True/0/"+metric+"_"+str(class_idx[dataset])+".png")
			gt = readimage("./../resnet/paper_images/selected/True/0/gt2d_"+str(class_idx[dataset])+".png")
		else:
			exp = readimage("./../natural/paper_images/selected/True/0/"+metric+"_"+str(class_idx[dataset])+".png")
			gt = readimage("./../natural/paper_images/selected/True/0/gt2d_"+str(class_idx[dataset])+".png")
		
		gt = np.round(gt, decimals=2)
		
		if metric == metrics[0]:
			if "synth" in dataset:
				input_img = cv2.imread("./../resnet/paper_images/selected/True/0/input_"+str(class_idx[dataset])+".png")
				cv2.imwrite("./real_world_synth_vis/input_synth.png",input_img)
				cv2.imwrite("./real_world_synth_vis/gt_synth.png",np.abs(gt)*255)
			else:
				input_img = cv2.imread("./../natural/paper_images/selected/True/0/input_"+str(class_idx[dataset])+".png")
				cv2.imwrite("./real_world_synth_vis/input_natural.png",input_img)
				cv2.imwrite("./real_world_synth_vis/gt_natural.png",np.abs(gt)*255)

		maskC = np.ones_like(gt)
		maskC[gt == 0] = 0 

		maskNC = np.zeros_like(gt)
		maskNC[gt == 0] = 1 

		cv2.imwrite("./real_world_synth_vis/"+dataset+"_"+metric+"_C.png", (maskC  * np.abs(exp)) * 255)

		cv2.imwrite("./real_world_synth_vis/"+dataset+"_"+metric+"_NC.png", (maskNC  * np.abs(exp)) * 255)


