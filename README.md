# Data-Science-Project-6th-sem-

Indian Liver Patient 

“Indian Liver Patient Dataset”

OVERVIEW


Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across.

As it is evident from the name, it gives the computer that which makes it more similar to humans: The ability to learn.

Machine learning is actively being used today, perhaps in many more places than one would expect.

The basic premise of machine learning is to build algorithms that can receive input data and use statistical analysis to predict an output while updating outputs as new data becomes available.



TYPES OF LEARNING:

1.	Supervised Learning

2.	Unsupervised Learning


ALGORITHMS


NAÏVE BAYES CLASSIFIER:

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable y and dependent feature vector x1 through xn, :
P(y∣x1,…,xn)=P(y)P(x1,…xn∣y)P(x1,…,xn)
Using the naive conditional independence assumption that
P(xi|y,x1,…,xi−1,xi+1,…,xn)=P(xi|y),
for all i, this relationship is simplified to
P(y∣x1,…,xn)=P(y)∏i=1nP(xi∣y)P(x1,…,xn)
Since P(x1,…,xn) is constant given the input, we can use the following classification rule:
P(y∣x1,…,xn)∝P(y)∏i=1nP(xi∣y)⇓y^=arg⁡maxyP(y)∏i=1nP(xi∣y),
and we can use Maximum A Posteriori (MAP) estimation to estimate P(y) and P(xi∣y); the former is then the relative frequency of class y in the training set.
The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi∣y).
In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They require a small amount of training data to estimate the necessary parameters. (For theoretical reasons why naive Bayes works well, and on which types of data it does, see the references below.)
Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
 
RANDOM FOREST CLASSIFIER:

Decision trees are a popular method for various machine learning tasks. Tree learning "comes closest to meeting the requirements for serving as an off-the-shelf procedure for data mining", because it is invariant under scaling and various other transformations of feature values, is robust to inclusion of irrelevant features, and produces inspect able models.



However, they are seldom accurate. In particular, trees that are grown very deep tend to learn highly irregular patterns: they over fit their training sets, i.e. have low bias, but very high variance.



Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance.



This comes at the expense of a small increase in the bias and some loss of interpretability, but generally greatly boosts the performance in the final model.
 
K- NEAREST NEIGHBOUR:

The training examples are vectors in a multidimensional feature space, each with a class label. The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples.



In the classification phase, k is a user-defined constant, and an unlabeled vector (a query or test point) is classified by assigning the label which is most frequent among the k training samples nearest to that query point.



A commonly used distance metric for continuous variables is Euclidean distance. For discrete variables, such as for text classification, another metric can be used, such as the overlap metric (or Hamming distance).



In the context of gene expression microarray data, for example, k-NN has also been employed with correlation coefficients such as Pearson and Spearman.



Often, the classification accuracy of k-NN can be improved significantly if the distance metric is learned with specialized algorithms such as Large Margin Nearest Neighbor or Neighborhood components analysis.



A drawback of the basic "majority voting" classification occurs when the class distribution is skewed. That is, examples of a more frequent class tend to dominate the prediction of the new example, because they tend to be common among the k nearest neighbors due to their large number.



One way to overcome this problem is to weight the classification, taking into account the distance from the test point to each of its k nearest neighbors. The class (or value, in regression problems) of each of the k nearest points is multiplied by a weight proportional to the inverse of the distance from that point to the test point.

Another way to overcome skew is by abstraction in data representation.




For example, in a self-organizing map (SOM), each node is a representative (a center) of a cluster of similar points, regardless of their density in the original training data. K-NN can then be applied to the SOM.
 


