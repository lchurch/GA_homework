from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

def load_iris_data() :

    iris = datasets.load_iris()

    return (iris.data, iris.target, iris.target_names)

def knn(X_train, y_train, k_neighbors = 3 ) :
# I want to be able to loop through different numbers of nearest neighbors
#I tried doing this (no luck):
# def knn(Xtrain, y_train, k_neighbors = nn):
    #for nn in range (1,51,2) :
    # clf = KNeighborsClassifier(k_neighbors)
    # clf.fit(X_train, y_train)
    # knnscore = clf.score(?, ?)
    # return clf
    #print "Neighbors: <<%s>>, Accuracy: <<%s>>" % (nn, knnscore)
#I also tried a few other functions like accuracy_score

        clf = KNeighborsClassifier(k_neighbors)
        clf.fit(X_train, y_train)

        return clf

def nb(X_train, y_train) :

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

def cross_validate(XX, yy, classifier, k_fold) :

    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    
    for train_slice, test_slice in k_fold_indices :

        model = classifier(XX[[ train_slice  ]],
                         yy[[ train_slice  ]])

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    return k_score_total/k_fold

print "In this exercise, we are attempting to build a model that accurately  predicts the class of records in the Iris Dataset using KNN and Naive Bayes classification methods.  The solution performs a cross-validation of both KNN and Naive Bayes classifiers for n = 2, 3 and 10.  It returns the accuracies of each fold and the value of n that produces the highest accuracy. Performing the cross-validation on the two types of models is important for preventing overfitting Selecting the best-performing  n  allows us to select the model with the lowest predicted OSS error rate. See the results below:"

(XX,yy,y)=load_iris_data()    

classifiers_to_cv=[("kNN",knn),("Naive Bayes",nb)]

for (c_label, classifier) in classifiers_to_cv :

    print "---> %s <---" % c_label
    
    best_k=0
    best_cv_a=0
    for k_f in [2,3,10] :
       cv_a = cross_validate(XX, yy, classifier, k_fold=k_f)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)

    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)

print "Based on these results, I would set n to 3."
