# BEAM: Exact Benchmarking of Explainable AI Attribution Methods

## Evaluation of XAI Methods

The folder *new metrics* contains our implementation of BEAM using the baseline model (see folder *simple*), the vision transformer model (see folder *vit*), and the ResNet50 model (see folder *resnet*). 

In all cases, the raw results and runtimes of XAI metrics are stored in the *table_results_True_<run id>.csv* and *table_times_True_<run id>.csv* tables. These can be converted into the format shown in our paper and supplement using the code in the *results* subfolder.

Examples of XAI method explanations are displayed in the *paper_images/selected/True/<run id>* folders.


## Evaluation of XAI Metrics

The folder *new columns* contains our implementation used to evaluate XAI metrics using the dataset associated with the baseline model (see folder *simple*), the vision transformer model (see folder *vit*), and the ResNet50 model (see folder *resnet*). 

In all cases, the raw results of the evaluation of XAI metrics are stored in the *table_results_True_<run id>_<gts turned binary>.csv* tables. These can be converted into the format shown in our paper and supplementary material using the code in the *results* subfolder.


## Evaluation of the Transferability of BEAM to Trained Models

The folder *new metrics/natural* contains our implementation of XAI metric evaluation using a model trained on a natural dataset.

The raw results and runtimes of XAI metrics are stored in the *table_results_True_<run id>.csv* and *table_times_True_<run id>.csv* tables. These can be converted into the format shown in our paper and supplement using the code in the *results* subfolder. *gen_tables.py* computes the tables, and *mad.py* produces the mean absolute difference plots and numbers with respect to the results of the ResNet50 model with manually set weights. 

Examples of XAI method explanations are displayed in the *paper_images/selected/True/<run id>* folders.

## Visualization of GT Construction

The folder *visualization of gt construction* contains the code used to create the visualizations of the intermediate steps of gt construction shown in our paper. 

## Visualization of Synthetic vs Natural Input Image Based Explanations

The folder *new metrics/visualization of natural vs synth explanations* contains the code used to create the visualizations of the difference between explanations of natural input images, and those of synthetic input images. The folder *real_world_synth_vis* contains the produced images also contained in the paper.



