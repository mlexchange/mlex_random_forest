from sklearn.ensemble import RandomForestClassifier
import argparse
import pathlib
import glob
import skimage
import numpy as np
import PIL.Image
import joblib

import json
from feature_generation import img_to_ubyte_array, multiscale_basic_features
from model_validation import TrainingParameters, Metadata

""" Train a random forest classifier
    Input: design matrix, labelled masks
    Output: trained model, its-a-pickle!

code from: https://github.com/plotly/dash-sample-apps/blob/d96997bd269deb4ff98b810d32694cc48a9cb93e/apps/dash-image-segmentation/trainable_segmentation.py#L130
"""

# def fit_segmenter(labels, features, clf):
#     """
#     Segmentation using labeled parts of the image and a classifier.
#     Parameters
#     ----------
#     labels : ndarray of ints
#         Image of labels. Labels >= 1 correspond to the training set and
#         label 0 to unlabeled pixels to be segmented.
#     features : ndarray
#         Array of features, with the first dimension corresponding to the number
#         of features, and the other dimensions correspond to ``labels.shape``.
#     clf : classifier object (a scikit model)
#         classifier object, exposing a ``fit`` and a ``predict`` method as in
#         scikit-learn's API, for example an instance of
#         ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
#     Returns
#     -------
#     output : ndarray
#         Labeled array, built from the prediction of the classifier trained on
#         ``labels``.
#     clf : classifier object
#         classifier trained on ``labels``
#     Raises
#     ------
#     NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
#     """
#     # training process
#     training_data = features[:, labels > 0].T
#     training_labels = labels[labels > 0].ravel()
#     clf.fit(training_data, training_labels)  
#     
#     # predicting process
#     data = features[:, labels == 0].T
#     predicted_labels = clf.predict(data)
#     
#     output = np.copy(labels)
#     output[labels == 0] = predicted_labels
#     
#     return output, clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # reading args for feature generation
    parser.add_argument('image_dir', help='image filepath')
    parser.add_argument('feature_dir', help='output filepath')
    
    # reading args for training
    parser.add_argument('mask_dir', help='path to mask directory')
    parser.add_argument('model_dir', help = 'path to model (output) directory')
    parser.add_argument('parameters', help='dictionary that contains training parameters')
    
    args = parser.parse_args()
    
    images_path = pathlib.Path(args.image_dir)
    feature_dir = pathlib.Path(args.feature_dir)
    mask_dir = pathlib.Path(args.mask_dir)
    model_dir = pathlib.Path(args.model_dir)

    ###INPUT_ARGS_HARDCORE
    feature_list = {'intensity': True,
                    'edges': False,
                    'texture': False}

    for im in images_path.glob('*.tif'):    # this only takes the labeled images (*_for_training.tif)
        im_name_root = im.name.strip(im.suffix)
        image = img_to_ubyte_array(im)
        features = multiscale_basic_features(
                image,
                multichannel=False,
                intensity=feature_list['intensity'],
                edges=feature_list['edges'],
                texture=feature_list['texture']
                )
        num_features = features.shape[0]
        feature_out_name = str(feature_dir / im_name_root)+'.feature'
        np.savetxt(feature_out_name, features.reshape(num_features,-1))
        print('features generated for: {}\n'.format(feature_out_name))


    f_list = [np.genfromtxt(f) for f in feature_dir.glob("*feature")]
    all_features =np.concatenate(f_list, axis=-1).T

    ### READ IN IMAGE LIST ###
    mask_list = [np.genfromtxt(im).ravel() for im in mask_dir.glob('n-*')]
    all_mask = np.concatenate(mask_list)


    ### CHECK THAT n_features == n_images
    assert(len(all_features) == len(all_mask))
    train_features = all_features[ all_mask>-1,:]
    train_mask = all_mask[all_mask > -1]

    # Load training parameters
    if args.parameters is not None:
        parameters = TrainingParameters(**json.loads(args.parameters))
        
    print(f'parameters.oob_score: {parameters.oob_score}\n')
    ### CREATE RANDOM FOREST CLF ###
    #clf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, max_depth=8, max_samples=0.05)
    clf = RandomForestClassifier(n_estimators=parameters.n_estimators, 
                                 oob_score=parameters.oob_score, n_jobs=-1, max_depth=parameters.max_depth, max_samples=0.05)

    clf.fit(train_features,train_mask)
    if parameters.oob_score:
        oob_error = 1 - clf.oob_score_
        header = list(Metadata.__fields__)
        with open('training_logs.txt','w') as f:
            f.write(",".join(header) + "\n")
            f.write(f'{oob_error}')
            f.close()
        
    model_output_name = model_dir / 'random-forest.model'
    joblib.dump(clf, model_output_name)
    print('trained random forest: {}\n'.format(model_output_name))
    

