#!/usr/bin/env python3

import argparse
import itertools
import numpy as np
import pickle
import lzma

import sklearn
import sklearn.metrics
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.neural_network
import sklearn.calibration

import matplotlib.pyplot as plt

import Dataset
from Transformers import RGB2GrayTransformer, ContrastTransformer


parser = argparse.ArgumentParser()
# These arguments are used to specify necessary program variables:
parser.add_argument("--data_size", default=None, type=int, help="Data size")
parser.add_argument("--test_size", default=None, type=int, help="Data size")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--dataset_path", default="cifar-100-python.tar.gz", type=str, help="Path to dataset.")
parser.add_argument("--classification_method", default="svm", type=str, help="Classification method ('svm', 'mlp', 'boost').")
parser.add_argument("--fine", default=False, action='store_true', help="Train fine classes (subclasses of choosen superclass).")
parser.add_argument("--predict", default=False, action='store_true', help="Run prediction on given data")
parser.add_argument("--plot", default=False, action='store_true', help="Plot the predictions")
parser.add_argument("--save_plot", default=None, type=str, help="Whether and where to save the plot (if `None` -> don't)")
parser.add_argument("--model_path", default="project.model", type=str, help="Where to save the model.")


def chooseClassifier(args):
    """
    Decides based on the program argumets which classifier and which parameters 
    for cross-validation to use.

    Return classifier object and dictionary of parameters for cross-validation. 
    """

    classifier = None
    grid_search_params = None

    if args.classification_method == "svm":
        # Support Vectors Machine classifier
        svm = sklearn.svm.LinearSVC(random_state=args.seed)
        # To possibility of getting probabilities of the classes
        classifier = sklearn.calibration.CalibratedClassifierCV(svm)

        # Parameters tolerance `tol` and `C` (regularization parameter).
        grid_search_params = {  "classifier__base_estimator__tol": [1e-5, 1e-4, 1e-3],
                                "classifier__base_estimator__C": [0.5, 1, 5]
        }

    elif args.classification_method == "mlp":
        # Multi Layer Perceptron classifier
        classifier = sklearn.neural_network.MLPClassifier(random_state=args.seed)

        # Parameters `hidden_layer_sizes` (shape of the hidden layer),
        # `alpha` (L2 penalty parameter) and `learning_rate_init` (initial learning rate).
        grid_search_params = {  "classifier__hidden_layer_sizes": [(20,), (50,)],
                                "classifier__alpha": [0.001, 0.002],
                                "classifier__learning_rate_init": [0.001, 0.003, 0.005]
        }

    elif args.classification_method == "boost":
        # Gradient Boosted Trees classifier
        classifier = sklearn.ensemble.GradientBoostingClassifier(random_state=args.seed)
        
        # Parameters `learning_rate` and `max_features` 
        # (number of feature to consider when choosing the best split).
        grid_search_params = {  "classifier__learning_rate": [0.05, 0.1, 0.2],
                                "classifier__max_features": [None, "auto"]
        }
    
    return classifier, grid_search_params


def plot_data(dataset, y_score, classifier_name, save_filename=None):
    """
    Function to plot the ROC curves of the test data predictions.
        - params:   `dataset` - working dataset
                    `y_score` - predicted probabilities
                    `classifier_name` - name of classifier (for label in plot)
                    `save_filename` - whether to save the plot and where (if `None` -> don't save)

    Code is inspired by the example from the documentation of:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    y = sklearn.preprocessing.label_binarize(dataset.test_target, 
                                            classes=np.arange(dataset.num_classes))
        
    # Trick for having 2 columns in binary classification
    # (binarizer in 2 classes creates matrix only with one column)
    # Create binarizer for 3 classes and throw away the last (not used).
    if dataset.num_classes == 2:
        y = sklearn.preprocessing.label_binarize(dataset.test_target, 
                                                    classes=[0,1,2])
        # Cut the last column.
        y = y[:,:2]
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(dataset.num_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(dataset.num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(dataset.num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC for computing macro-average
    mean_tpr /= dataset.num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue", "magenta", "chocolate"])
    for i, color in zip(range(dataset.num_classes), colors):
        # for i in range(dataset.num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            # lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve of {classifier_name} classifier")
    plt.legend(loc="lower right")

    if save_filename != None:
        plt.savefig(save_filename)

    else:
        plt.show()



def main(args):
    # Prepare dataset
    dataset = None
    if args.fine:
        # Learning fine classes.
        dataset = Dataset.Dataset(args, args.dataset_path, superclassFine=True)
    else:
        # Learning superclasses.
        dataset = Dataset.Dataset(args, args.dataset_path)


    # Take only subset of the data (for testing the program to run in reasonable time).
    # Subset of the Train Data
    if args.data_size != None:
        dataset.train_data = dataset.train_data[:args.data_size]
        dataset.train_target = dataset.train_target[:args.data_size]
    
    # Subset of the Test Data
    if args.test_size != None:
        dataset.test_data = dataset.test_data[:args.test_size]
        dataset.test_target = dataset.test_target[:args.test_size]


    # Choose whether predict values or train model.
    if not args.predict:
        # Train Model

        # Get correct classifier and parameters for cross-validation.
        classifier, grid_search_params = chooseClassifier(args)
    

        # Add transformers and classifier into one pipeline 
        # (first performs preprocessing then starts classifier).
        model = sklearn.pipeline.Pipeline([
            # From RGB to gray scale
            ("RGB_to_gray", RGB2GrayTransformer()),
            # Improves contranst of the image
            ("contrast", ContrastTransformer()),
            # Standardize features (mean=0, variance=1)
            ("scale", sklearn.preprocessing.StandardScaler()),
            ("classifier", classifier)
        ])

        # Add to model cross-validation 
        # (it trains on different parameters and choosed the best model
        #  which will be stored in the model variable)
        model = sklearn.model_selection.GridSearchCV(
            model,
            grid_search_params,
            cv=sklearn.model_selection.StratifiedKFold(3)   # stratified 
        )

        # Train the model
        model.fit(dataset.train_data, dataset.train_target)

        # Print the grid search results (to check best parameters).
        for rank, accuracy, params in zip(model.cv_results_["rank_test_score"], model.cv_results_["mean_test_score"], model.cv_results_["params"]):
            print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy), *("{}: {}".format(key, value) for key, value in params.items()))
        
        
        test_accuracy = sklearn.metrics.accuracy_score(dataset.test_target, model.predict(dataset.test_data))
        print(test_accuracy)

        # Save the trained model to specified file.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
    
    
    else:
        # Prediction

        # Load the trained model.
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Prediction of the classes
        prediction = model.predict(dataset.test_data)
        # Prediction of the classes probabilities
        y_score = model.predict_proba(dataset.test_data)

        # Computation of the confusion matrix and the macro-averaged precision and recall:
        confusion_matrix = sklearn.metrics.confusion_matrix(dataset.test_target, prediction)
        precision = sklearn.metrics.precision_score(dataset.test_target, prediction, average="macro")
        recall = sklearn.metrics.recall_score(dataset.test_target, prediction, average="macro")

        print(f"Confusion matrix: \n{confusion_matrix}")
        print(f"Macro-averaged precision: {precision}")
        print(f"Macro-averaged recall: {recall}")

        # Ploting ROC curve if we want
        if args.plot:
            # Get name of the plot header
            clf_name = None
            if args.classification_method == "svm":
                clf_name = "SVM"
            elif args.classification_method == "mlp":
                clf_name = "MLP"
            elif args.classification_method == "boost":
                clf_name = "Gradient Boosted Trees"

            plot_data(dataset, y_score, clf_name, args.save_plot)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)