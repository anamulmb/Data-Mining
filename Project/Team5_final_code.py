#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


home_dir=os.getcwd()

data_dir = os.path.join(home_dir,"roiData")
data_dir2 =  os.path.join(home_dir,"func_preproc")
home_dir,data_dir,data_dir2


# In[ ]:


# rois_dir = os.path.join(data_dir, 'data', 'rois')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# In[ ]:


if not os.path.exists(data_dir2):
    os.makedirs(data_dir2)


# In[ ]:


from nilearn import datasets
from nilearn import decomposition
from nilearn import plotting,image

import numpy as np

from nilearn.input_data import NiftiMapsMasker

from nilearn.regions import RegionExtractor

from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt


# In[ ]:


# for fetching dataset for first instance only
abide = datasets.fetch_abide_pcp(data_dir= data_dir2,derivatives=['func_preproc'],
                        SITE_ID=['NYU'],
#                         n_subjects=3
                                )
print(abide.keys())


func = abide.func_preproc


# In[ ]:


type(func)


# In[ ]:


def get_key(filename):
    f_split = filename.split('_')
#     print(f_split)
    if f_split[2] == 'func':
        key = '_'.join(f_split[0:2]) 
    else:
        key = ""
    return key


# In[ ]:


fetched_data = os.path.join(data_dir2,"ABIDE_pcp")
fetched_data


# In[ ]:


flist = os.listdir(fetched_data)
flist2 = os.listdir(os.path.join(fetched_data,flist[0]))
flist3 =  os.listdir(os.path.join(fetched_data,flist[0],flist2[0]))
flist,flist2,flist3
flist3[0].split('_')


# In[ ]:


func


# In[ ]:


prefix_path = os.path.join(fetched_data,flist[0],flist2[0])
prefix_path+"\\"+flist3[0]


# In[ ]:


func2 = []
for filename in flist3:
    func2.append(prefix_path+"\\"+filename)


# In[ ]:



# use it if not downloading the dataset
# func2 == func
# if (func2==func):
#     func=func2


# In[ ]:


# import nibabel as nib

# func_nib = nib.load(func2[0])
# print(func_nib)


# In[ ]:


func_nib.shape,func_nib.get_fdata()


# In[ ]:





# In[ ]:





# In[ ]:


print(len(flist3))
file_key_list=[]
for f in range(len(flist3)):
    file_key_list.append(get_key(flist3[f]))
# file_key_list


# In[ ]:





# In[ ]:


import pandas as pd

df_labels = pd.read_csv(os.path.join(fetched_data,'Phenotypic_V1_0b_preprocessed1.csv'))#path 
# print(df_labels)

df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2:0})
# print(len(df_labels))


# In[ ]:


labels = {}
for row in df_labels.iterrows():
    file_id = row[1]['FILE_ID']
    y_label = row[1]['DX_GROUP']
    if file_id == 'no_filename':
        continue
    assert(file_id not in labels)
    labels[file_id] = y_label


# In[ ]:


# labels
# ,file_key_list


# In[ ]:



def get_label(filename):
    assert (filename in labels)
    return labels[filename]


# In[ ]:


# Method 1
# CANICA

canica = decomposition.CanICA(n_components=20,
                              mask_strategy='background',
                              n_init=10)
canica.fit(func)


# In[ ]:


components = canica.components_img_
# Use CanICA's masker to project the components back into 3D space
# components_img = canica.masker_.inverse_transform(components)


# In[ ]:





# In[ ]:


# the following line: saves components in file
components.to_filename('canica_resting_state.nii.gz')


# In[ ]:


from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(components, title='All ICA components')


# In[ ]:


from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(components)):
    plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                  cut_coords=1, colorbar=False)


# In[ ]:


from nilearn.decomposition import DictLearning

dict_learning = DictLearning(n_components=20,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             random_state=0,
                             n_epochs=1,
                             mask_strategy='background')

print('[Example] Fitting dicitonary learning model')
dict_learning.fit(func)
print('[Example] Saving results')
# Grab extracted components umasked back to Nifti image.
# Note: For older versions, less than 0.4.1. components_img_
# is not implemented. See Note section above for details.
dictlearning_components_img = dict_learning.components_img_
dictlearning_components_img.to_filename('dictionary_learning_resting_state.nii.gz')


# In[ ]:


plot_prob_atlas(dictlearning_components_img,
                title='All DictLearning components')


# In[ ]:


for i, cur_img in enumerate(iter_img(dictlearning_components_img)):
    plot_stat_map(cur_img, display_mode="z", title="Comp %d" % i,
                  cut_coords=1, colorbar=False)


show()


# In[ ]:





# In[ ]:


components.shape


# In[ ]:


# Use CanICA's masker to project the components back into 3D space
components = canica.components_
components_img = canica.masker_.inverse_transform(components)


# In[ ]:



masker = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01)


# In[ ]:


subjects_timeseries = {}
for subject_func in func:
#     print (subject_func.split('\\'))
    key = get_key(subject_func.split('\\')[-1])
    subjects_timeseries[key] = masker.fit_transform(subject_func)

#     subjects_timeseries.append(masker.fit_transform(subject_func))
    
#     
# Visualizing extracted timeseries signals. We import matplotlib.pyplot
import matplotlib.pyplot as plt


# In[ ]:


subjects_timeseries


# In[ ]:


correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrices = correlation_measure.fit_transform(subjects_timeseries.values())
key_indices = subjects_timeseries.keys()
# key_indices
# for i in subjects_timeseries:
#     correlation_matrices[i] = correlation_measure.fit_transform(subjects_timeseries[i])


# In[ ]:


correlation_matrices.shape


# In[ ]:


# labels


# In[ ]:


# abide_correlations = []
# control_correlations = []
# for i in subjects_timeseries:
#     print(labels[i])
#     if labels[i] == 1:
#         abide_correlations.append(correlation_matrices[i])
#     else:
#         control_correlations.append(correlation_matrices[i])


# In[ ]:


# region extraction
extractor = RegionExtractor(components_img, threshold=2.,
                            thresholding_strategy=
                            'ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True,
                            min_region_size=1350)


extractor.fit()

regions_extracted_img = extractor.regions_img_


n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img,
                         view_type='filled_contours',
                         title=title, threshold=0.008)


# In[ ]:


# dictionary learning method2
# dictlearning_components_img



extractor = RegionExtractor(dictlearning_components_img, threshold=2.,
                            thresholding_strategy=
                            'ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True,
                            min_region_size=1350)


extractor.fit()

regions_extracted_img = extractor.regions_img_


n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img,
                         view_type='filled_contours',
                         title=title, threshold=0.008)


# In[ ]:


correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrices = correlation_measure.fit_transform(subjects_timeseries.values())
subject_id = list(subjects_timeseries.keys())
subject_id[0]


# In[ ]:


abide_correlations = []
control_correlations = []
for i in subjects_timeseries.keys():
#     print(i)
#     print(subject_id.index(i))
    if labels[i] == 1:
        abide_correlations.append(correlation_matrices[subject_id.index(i)])
    else:
        control_correlations.append(correlation_matrices[subject_id.index(i)])


# In[ ]:


#Getting the mean correlation matrix across all treatment subjects
mean_correlations_abide = np.mean(abide_correlations, axis=0).reshape(subjects_timeseries[subject_id[0]].shape[-1],
                                                          subjects_timeseries[subject_id[0]].shape[-1])
#Getting the mean correlation matrix across all control subjects
mean_correlations_control = np.mean(control_correlations, axis=0).reshape(subjects_timeseries[subject_id[0]].shape[-1],
                                                          subjects_timeseries[subject_id[0]].shape[-1])
#Visualizing the mean correlation
plotting.plot_matrix(mean_correlations_abide, vmax=1, vmin=-1,
                               colorbar=True, title='Correlation between 20 regions for Abide')
plotting.plot_matrix(mean_correlations_control, vmax=1, vmin=-1,
                               colorbar=True, title='Correlation between 20 regions for controls')


# In[ ]:


#Getting the center coordinates from the component decomposition to use as atlas labels
coords = plotting.find_probabilistic_atlas_cut_coords(components_img)
#Plotting the connectome with 80% edge strength in the connectivity
plotting.plot_connectome(mean_correlations_abide, coords,
                         edge_threshold="80%", title='Correlation between 20 regions for Abide')
plotting.plot_connectome(mean_correlations_control, coords,
                         edge_threshold="80%", title='Correlation between 20 regions for controls')
plotting.show()


# In[ ]:


correlation_matrices.shape


# In[ ]:





# In[ ]:


# prev_path = os.path.join(fetched_data,flist[0],flist2[0])
# file_names = os.listdir(prev_path)

# full_file_names = []
# for i in file_names:
#     full_file_names.append(prev_path+"\\"+i)
    
# full_file_names

# from nilearn.connectome import ConnectivityMeasure

# correlations = []

# # Initializing ConnectivityMeasure object with kind='correlation'
# connectome_measure = ConnectivityMeasure(kind='correlation')
# for filename in subjects_timeseries:
#     print (filename)
# #     # call transform from RegionExtractor object to extract timeseries signals
#     timeseries_each_subject = extractor.transform(full_file_names)
# #     # call fit_transform from ConnectivityMeasure object
#     correlation = connectome_measure.fit_transform([timeseries_each_subject])
# #     # saving each subject correlation to correlations
#     correlations.append(correlation)

# # Mean of all correlations
# # import numpy as np
# # mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
# #                                                           n_regions_extracted)


# In[ ]:



from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from nilearn.connectome import sym_matrix_to_vec


# In[ ]:


# n_splits=10 default for 10 fold cross validation
str_shuffle_spt = StratifiedShuffleSplit(n_splits=10,test_size=.2,random_state=0)


# In[ ]:



stratified_shuffle_split = StratifiedShuffleSplit(test_size=.2)


measures = ['correlation', 'partial correlation', 'tangent']
predictors = [
    ('svc_l2', LinearSVC(C=1)),
    ('svc_l1', LinearSVC(C=1, penalty='l1', dual=False)),
    ('ridge_classifier', RidgeClassifier()),
]


# In[ ]:


y=[]
for i in subjects_timeseries.keys():
    y.append(labels[i])
type(y)


# In[ ]:





for measure in measures:
    conn_est = ConnectivityMeasure(kind=measure)
    conn_matrices = conn_est.fit_transform(subjects_timeseries.values())
    X = sym_matrix_to_vec(conn_matrices)
    
    for name, predictor in predictors:
#         print (measure)
#         print (name)
        print(measure, name, np.mean(cross_val_score(predictor, X, np.array(y), cv=stratified_shuffle_split.split(X,y))))
#       print( cross_val_score(predictor, X, np.array(y), cv=stratified_shuffle_split.split(X,y)))
      
#         classifier = predictor
#         for train_index,test_index in stratified_shuffle_split.split(X,y):
#             predictor.fit(X[train_index],y[train_index])
#             y_score = predictor.decision_function(X[test_index])
#             average_precision = average_precision_score(y[test_index], y_score)
#             print('Average precision-recall score: {0:0.2f}'.format(
#                 average_precision))
#             disp = plot_precision_recall_curve(predictor, X[test_index], y[test_index])
#             disp.ax_.set_title('2-class Precision-Recall curve: '
#                                'AP={0:0.2f}'.format(average_precision))
#             print(y_score)


# In[ ]:


y.shape,X.shape


# In[ ]:


y=np.array(y)
for measure in measures:
    conn_est = ConnectivityMeasure(kind=measure)
    conn_matrices = conn_est.fit_transform(subjects_timeseries.values())
    X = sym_matrix_to_vec(conn_matrices)
    
    for name, predictor in predictors:
        test_set = []
        for train_index,test_index in stratified_shuffle_split.split(X,y):
            predictor.fit(X[train_index],y[train_index])
            test_set.append(test_index)
        
        print(np.concatenate)
#          10 graphs for each case
        for i in test_set:
            y_score = predictor.decision_function(X[test_index])
            average_precision = average_precision_score(y[test_index], y_score)
            
        print(name)
        print('Average precision-recall for '+measure+" "+name +' score: {0:0.2f}'.format(
                average_precision))
        disp = plot_precision_recall_curve(predictor, X[test_index], y[test_index])
        disp.ax_.set_title('2-class Precision-Recall for('+measure+", "+name+') curve: '
                               'AP={0:0.2f}'.format(average_precision))
#             print(y_score)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




