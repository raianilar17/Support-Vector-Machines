# Support-Vector-Machines

Build a spam classifier Using support vector machines (SVMs)

This code is successfully implemented on octave version 4.2.1

To get started with the project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave to change to this directory before starting this exercise.

This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

# Environment Setup Instructions

## Instructions for installing Octave 

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

# Files included in this Project

[ex6.m](ex6.m) - Octave script for the first half of the project

[ex6data1.mat](ex6data1.mat) - Example Dataset 1

[ex6data2.mat](ex6data2.mat) - Example Dataset 2

[ex6data3.mat](ex6data3.mat) - Example Dataset 3

[svmTrain.m](svmTrain.m ) - SVM training function

[svmPredict.m](svmPredict.m) - SVM prediction function

[plotData.m](plotData.m) - Plot 2D data

[visualizeBoundaryLinear.m](visualizeBoundaryLinear.m) - Plot linear boundary

[visualizeBoundary.m](visualizeBoundary.m) - Plot non-linear boundary

[linearKernel.m](linearKernel.m) - Linear kernel for SVM

[gaussianKernel.m](gaussianKernel.m) - Gaussian kernel for SVM

[dataset3Params.m](dataset3Params.m) - Parameters to use for Dataset 3

[ex6_spam.m](ex6_spam.m) - Octave/MATLAB script for the second half of the project

[spamTrain.mat](spamTrain.mat) - Spam training set

[spamTest.mat](spamTest.mat) - Spam test set

[emailSample1.txt](emailSample1.txt) - Sample email 1

[emailSample2.txt](emailSample2.txt) - Sample email 2

[spamSample1.txt](spamSample1.txt) - Sample spam 1

[spamSample2.txt](spamSample2.txt) - Sample spam 2

[vocab.txt](vocab.txt) - Vocabulary list

[getVocabList.m](getVocabList.m) - Load vocabulary list

[porterStemmer.m](porterStemmer.m)- Stemming function

[readFile.m](readFile.m) - Reads a file into a character string

[processEmail.m](processEmail.m) - Email preprocessing

[emailFeatures.m](emailFeatures.m) - Feature extraction from emails

Throughout the project, I use the script [ex6.m](ex6.m). These scripts set up the dataset for the problems and make calls to functions.

# Support Vector Machines

In the first half of this project, I used support vector machines(SVMs) with various example 2D datasets. Experimenting with these datasets will help me gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs. In the next half of the project, I used support vector machines to build a spam classifier.

The script, [ex6.m](ex6.m), will help you step through the first half of the project.

## Example Dataset 1

I began by with a 2D example dataset which can be separated by a linear boundary. The script [ex6.m](ex6.m) will plot the training data (Figure 1). In this dataset, the positions of the positive examples (indicated with +) and the negative examples (indicated with o) suggest a natural separation indicated by the gap. However, notice that there is an outlier positive example + on the far left at about (0.1, 4.1). As part of this project, I also see how this outlier affects the SVM decision boundary.

Figure 1: Example Dataset 1

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_1.png)


In this part of the project, I tried using different values of the C parameter with SVMs. Informally, the C parameter is a positive value that controls the penalty for misclassified training examples. A large C parameter tells the SVM to try to classify all the examples correctly. C plays a role similar to 1/λ  , where λ is the regularization parameter that I used previously for [logistic regression project](https://github.com/raianilar17/Logistic-Regression).

Figure 2: SVM Decision Boundary with C = 1 (Example Dataset 1)

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_2.png)

Figure 3: SVM Decision Boundary with C = 50 (Example Dataset 1)

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_3.png)

Figure 4: SVM Decision Boundary with C = 100 (Example Dataset 1)

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_4.png)

Figure 5: SVM Decision Boundary with C = 1000 (Example Dataset 1)

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_5.png)


The next part in [ex6.m](ex6.m) will run the SVM training (with C = 1) using SVM software that I have included with code, [svmTrain.m](svmTrain.m).In order to ensure compatibility with Octave/MATLAB, I included this implementation of an SVM learning algorithm. However, this particular implementation was chosen to maximize compatibility, and is not very efficient. If you are training an SVM on a real problem, especially if you need to scale to a larger dataset, I strongly recommend
instead using a highly optimized SVM toolbox such as [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [SVMLight](http://svmlight.joachims.org/).


When C = 1,I find that the SVM puts the decision boundary in the gap between the two datasets and misclassifies the data point on the far left (Figure 2).


## Implementation Note:

Most SVM software packages (including [svmTrain.m](svmTrain.m)) automatically add the extra feature(x_0 = 1) for you and automatically take care of learning the intercept term Theta_zero(θ_0). So when passing your training data to the SVM software, there is no need to add this extra feature x_0 = 1 yourself. In particular, In Octave/MATLAB your code should be working with training examples x ∈ R^n (rather than x ∈ R^n+1 ); for example, In the first example dataset x ∈ R^2 .

I tried different values of C on this dataset. Specifically,I change the value of C in the script to C = 50,100,1000 and run the SVM training again. When C = 100, I find that the SVM now classifies every single example correctly, but has a decision boundary that does not appear to be a natural fit for the data (Figure 4).The result of C=50 is in Figure 3 and C=1000 in Figure 5.

## SVM with Gaussian Kernels






