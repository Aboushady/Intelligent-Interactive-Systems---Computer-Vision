import os
import numpy as np
import cv2

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC


req_labels = [
    'Outer left eyebrow', 
    'Middle left eyebrow', 
    'Inner left eyebrow', 
    'Inner right eyebrow', 
    'Middle right eyebrow', 
    'Outer right eyebrow', 
    'Outer left eye corner', 
    'Inner left eye corner', 
    'Inner right eye corner', 
    'Outer right eye corner', 
    'Nose saddle left', 
    'Nose saddle right', 
    'Left nose peak', 
    'Nose tip', 
    'Right nose peak', 
    'Left mouth corner', 
    'Upper lip outer middle', 
    'Right mouth corner', 
    'Upper lip inner middle', 
    'Lower lip inner middle', 
    'Lower lip outer middle', 
    'Chin middle'
]


def parse_file(file_name, images, lm_coords):
    print(file_name)
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f]
        if lines[2] == "22 landmarks" and lines[5:27] == req_labels:
            img_fn = file_name[:-3] + "png"
            img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
            img_res = cv2.resize(img, (480, 640))
            images.append(img_res)
            h, w = img.shape
            h_rat, w_rat = 640/h, 480/w
            coords = [(int(float(x) * w_rat), int(float(y) * h_rat)) for x, y in [s.split(" ") for s in lines[29:51]]]
            # Some coordinates are out of bound for some reason, this fixes them
            coords = [(x if x >=   0 else   0, y if y >=   0 else   0) for x, y in coords]
            coords = [(x if x <= 479 else 479, y if y <= 639 else 639) for x, y in coords]
            lm_coords.append(np.array(coords))


def get_files(path):
    images, lm_coords = [], []
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(subdir, file)
            if file_name[-3:] == "lm2":
                parse_file(file_name, images, lm_coords)
                count += 1
        if count >= 100:
            break
    return np.array(images), np.array(lm_coords)


def write_images(images, lm_coords):
    for i in range(len(images)):
        img = images[i]
        coords = lm_coords[i]
        for x, y in coords:
            img[y, x] = 255
        cv2.imwrite("./images/img_{}.png".format(i), img)


def dim_reduce(data, use_tsne=False):
    X_trans = None
    # Data Reduction 
    if not use_tsne:
        # Dimensionality Reduction PCA 
        pca = PCA(n_components=2)
        X_trans = pca.fit_transform(data)
    else:
        # Manifold embedding with tSNE
        tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
        X_trans = tsne.fit_transform(data)
    return X_trans

                
def get_train_test(fnData, fnTargets, nSamples, percentSplit=0.7):
    trainData, testData, trainLabels, expectedLabels = train_test_split(fnData,
                                                                        fnTargets,
                                                                        test_size=(1.0-percentSplit),
                                                                        random_state=0)
    return trainData, trainLabels, testData, expectedLabels


def main():
    bos_path = "../BosphorusDB/bosphorusDB/__files__/__others__/BosphorusDB"
    
    # Get images and target coordinates from BosphorusDB to train on
    images, target_lm_coords = get_files(bos_path)
    
    n_samples = len(images)
    data = images.reshape((n_samples, -1))
    targets = target_lm_coords.reshape((n_samples, -1))
    
    # Trying to get 1D labels for all samples
    #targets = np.array(["".join(map(str, coords)) for coords in targets])
    
    # Dimensionality reduction
    # Should we perform normalization also?
    data = dim_reduce(data)
    
    
    
    # -------- TRAIN AND TEST --------
    
    X_train, X_labels, X_test, X_trueLabels = get_train_test(data, targets, n_samples)
    print(data.shape)
    print(X_labels.shape)
    print(X_trueLabels.shape)
    
    # k-NearestNeighbour (k-NN)
    n_neighbors = 10
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
    kNNClassifier.fit(X_train, X_labels)
    predictedLabels = kNNClassifier.predict(X_test)
    
    # Support Vector Machine (SVM)
    clf_svm = LinearSVC()
    clf_svm.fit(X_train, X_labels)
    y_pred_svm = clf_svm.predict(X_test)
    


    # -------- ANALYZE PERFORMANCE --------
    
    # k-NearestNeighbour (k-NN)
    print("\nClassification report for classifier k-NearestNeighbour:")
    print(metrics.classification_report(X_trueLabels, predictedLabels))
    print("\nConfusion matrix:")
    print(metrics.confusion_matrix(X_trueLabels, predictedLabels))
    # k-NN Cross Validation
    knn_scores = cross_val_score(kNNClassifier, data, targets, cv=5)
    print("\nkNN cross validation score:")
    print(knn_scores)
    
    # Support Vector Machine (SVM)
    print("\nLinear SVM accuracy:")
    print(metrics.accuracy_score(X_trueLabels, y_pred_svm))
    # SVM Cross Validation
    svm_scores = cross_val_score(clf_svm, data, targets, cv=5)
    print("\nkNN cross validation score:")
    print(svm_scores)



if __name__ == "__main__":
    main()