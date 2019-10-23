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

In this part of the project, I used SVMs to do non-linear classification. In particular, I used SVMs with Gaussian kernels on datasets that are not linearly separable.

## Gaussian Kernel

To find non-linear decision boundaries with the SVM, we need to first implement a Gaussian kernel. You can think of the Gaussian kernel as a similarity function that measures the “distance” between a pair of examples,(x^(i) , x^(j) ). The Gaussian kernel is also parameterized by a bandwidth parameter, σ, which determines how fast the similarity metric decreases (to 0) as the examples are further apart.

I wrote the code in [gaussianKernel.m](gaussianKernel.m) to compute the Gaussian kernel between two examples, (x^(i),x^(j)). The Gaussian kernel function is defined as:

K_gaussian (x^(i) , x^(j)) = exp(- || x^(i) - x^(j)||^2/(2*(σ^2)));

                           = exp(- sum_{k=1}^{n}((x^(i)_k - x^(j)_k)^2/(2*(σ^2)));
                           
 where :
 
 x^(i) = vector or matrix
 
 sum_{k=1}^{n} = summation of k = 1 to n
 
 σ = sigma
 
 After completion the function [gaussianKernel.m](gaussianKernel.m), the script [ex6.m](ex6.m) will test gaussian kernel      function on two examples and see a value of 0.324652.
 
 ## Example Dataset 2
 
 Figure 6: Example Dataset 2
 
 ![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_6.png)
 
The next part in [ex6.m](ex6.m) will load and plot dataset 2 (Figure 6). From the figure,I obserse that there is no linear decision boundary that separates the positive and negative examples for this dataset. However, by using the Gaussian kernel with the SVM, I will be able to learn a non-linear decision boundary that can perform reasonably well for the dataset.
 
 In the next part of [ex6.m](ex6.m) will proceed to train the SVM with the Gaussian kernel on this dataset.
 
 Figure 7: SVM (Gaussian Kernel) Decision Boundary (Example Dataset 2)
 
 ![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_7.png)

Figure 7 shows the decision boundary found by the SVM with a Gaussian kernel. The decision boundary is able to separate most of the positive and negative examples correctly and follows the contours of the dataset well.

## Example Dataset 3

In this part of the project, I gain more practical skills on how to use a SVM with a Gaussian kernel. The next part of [ex6.m](ex6.m) will load and display a third dataset (Figure 8). I used the SVM with the Gaussian kernel with this dataset.


Figure 8: Example Dataset 3

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_8.png)

In the dataset, [ex6data3.mat](ex6data3.mat),the variables are X,y, Xval, yval. The code in [ex6.m](ex6.m) trains the SVM classifier using the training set (X, y) using parameters loaded from [dataset3Params.m](dataset3Params.m).

I used the cross validation set Xval, yval to determine the best C and σ parameter to use.  For both C and σ, I tried values in multiplicative steps (e.g., 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30). Note that I try all possible pairs of values for C and σ (e.g., C = 0.3 and σ = 0.1). For example, if I try each of the 8 values listed above for C and for σ^2 , I end up training and evaluating (on the cross validation set) a total of 8^2 = 64 different models.

After I determined the best C and σ parameters , I modify the code in [dataset3Params.m](dataset3Params.m), filling in the best parameters I found. For my best parameters, the SVM returned a decision boundary shown in Figure 9.

Figure 9: SVM (Gaussian Kernel) Decision Boundary (Example Dataset 3)

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/FIGURE_9.png)

## Implementation Tip: 

When implementing cross validation to select the best C and σ parameter to use, you need to evaluate the error on the cross validation set. Recall that for classification, the error is defined as the fraction of the cross validation examples that were classified incorrectly. In Octave/MATLAB, you can compute this error using mean(double(predictions ~= yval)), where predictions is a vector containing all the predictions from the SVM, and yval are the true labels from the cross validation set. You can use the [svmPredict](svmPredict.m) function to generate the predictions for the cross validation set.

The ouput of [ex6.m](ex6.m) octave script

Type on octave Cli 

ex6

[output Looks like](output_ex6.txt)

# Spam Classification

Many email services today provide spam filters that are able to classify emails into spam and non-spam email with high accuracy. In this part of the project, I use SVMs to build my own spam filter.

I train a classifier to classify whether a given email, x, is spam (y = 1) or non-spam (y = 0). In particular, I need to convert each email into a feature vector x ∈ R^n . The following parts of the project will walk through how such a feature vector can be constructed from an email.

Throughout the rest of this project, I used the script [ex6_spam.m](ex6_spam.m). The dataset included for this project is based on a subset of the [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/).  For the purpose of this project, I only be use the body of the email (excluding the email headers) but during testing model , I used email with headers also.

## Preprocessing Emails

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/ex6_spam_figure1.png)

Figure 10 Shows sample email(Raw email contents), Before stem the word and After preprocessing and processed email

Before starting on a machine learning task, it is usually insightful to take a look at examples from the dataset. Figure 10 shows a sample email(raw email contents) that contains a URL, an email address (at the end), numbers, and dollar amounts. While many emails would contain similar types of entities (e.g.,numbers, other URLs, or other email addresses), the specific entities (e.g.,
the specific URL or specific dollar amount) will be different in almost every email. Therefore, one method often employed in processing emails is to “normalize” these values, so that all URLs are treated the same, all numbers are treated the same, etc. For example, we could replace each URL in the email with the unique string “httpaddr” to indicate that a URL was present.


This has the effect of letting the spam classifier make a classification decision based on whether any URL was present, rather than whether a specific URL was present. This typically improves the performance of a spam classifier, since spammers often randomize the URLs, and thus the odds of seeing any particular URL again in a new piece of spam is very small.

In [processEmail.m](processEmail.m), I implemented the following email preprocessing and normalization steps:

1. Lower-casing: The entire email is converted into lower case, so that captialization is ignored (e.g., IndIcaTE is treated                  the same as Indicate).

2. Stripping HTML: All HTML tags are removed from the emails. Many emails often come with HTML formatting; we remove all the
                   HTML tags, so that only the content remains.

3. Normalizing URLs: All URLs are replaced with the text “httpaddr”.

4. Normalizing Email Addresses: All email addresses are replaced with the text “emailaddr”.

5. Normalizing Numbers: All numbers are replaced with the text “number”.

6. Normalizing Dollars: All dollar signs ($) are replaced with the text "dollar".

7. Word Stemming: Words are reduced to their stemmed form. For example, “discount”, “discounts”, “discounted” and                             “discounting” are all replaced with “discount”. Sometimes, the Stemmer actually strips off additional                       characters from the    end, so “include”, “includes”, “included”, and “including” are all replaced with                     “includ”.

8. Removal of non-words: Non-words and punctuation have been removed. All white spaces (tabs, newlines, spaces) have all                              been trimmed to a single space character.


The result of these preprocessing steps is shown in Figure 10(processed email). While preprocessing has left word fragments and non-words, this form turns out to be much easier to work with for performing feature extraction.


## Vocabulary List

After preprocessing the emails, I have a list of words (e.g., Figure 10(processed email)) for each email. The next step is to choose which words I would like to use in our classifier and which I would want to leave out.

For this segment, I have chosen only the most frequently occuring words as our set of words considered (the vocabulary list). Since words that occurrarely in the training set are only in a few emails, they might cause the model to overfit our training set. The complete vocabulary list is in the file [vocab.txt](vocab.txt). Our vocabulary list was selected by choosing all words which occur at least a 100 times in the spam corpus, resulting in a list of 1899 words. In practice, a vocabulary list with about 10,000 to 50,000 words is often used.

Given the vocabulary list,I now map each word in the preprocessed emails (e.g., Figure 10) into a list of word indices that contains the index of the word in the vocabulary list. Figure 11 shows the mapping for the sample email. Specifically, in the sample email, the word “anyone” was first normalized to “anyon” and then mapped onto the index 86 in the vocabulary
list.

Figure 11: Word Indices for Sample Email

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/ex6_spam_figure2.png)

I wrote the code in [processEmail.m](processEmail.m) to perform this mapping. In the code, A string str which is a single word from the processed email. Then, I look up the word in the vocabulary list vocabList and find if the word exists in the vocabulary list. If the word exists, I add the index of the word into the word indices variable. If the word does not exist, and is therefore not in the vocabulary, I skip the word.

### Octave/MATLAB Tip:

In Octave/MATLAB, you can compare two strings with the strcmp function. For example, strcmp(str1, str2) will return 1 only when both strings are equal. In the provided starter code, vocabList is a “cell-array” containing the words in the vocabulary. In Octave/MATLAB, a cell-array is just like a normal array (i.e., a vector), except that its elements can also be strings (which they can’t in a normal Octave/MATLAB matrix/vector), and you index into them using curly braces instead of square brackets. Specifically, to get the word at index i, you can use vocabList{i}. You can also use length(vocabList) to get the number of words in the vocabulary.

## Extracting Features from Emails

Now, I implement the feature extraction that converts each email into a vector in R^n . For this part,I used n = # words in vocabulary list. Specifically, the feature x_i ∈ {0, 1} for an email corresponds to whether the i-th word in the dictionary occurs in the email. That is, x_i = 1 if the i-th word is in the email and x_i = 0 if the i-th word is not present in the email.


I wrote the code in [emailFeatures.m](emailFeatures.m) to generate a feature vector for an email, given the word indices.The next part of [ex6_spam.m](ex6_spam.m) will run code on the email sample and see that the feature vector had length 1899 and 45 non-zero entries.


## Training SVM for Spam Classification

The next step of [ex6_spam.m](ex6_spam.m) will load a preprocessed training dataset that will be used to train a SVM classifier. [spamTrain.mat](spamTrain.mat) contains 4000 training examples of spam and non-spam email, while [spamTest.mat](spamTest.mat) contains 1000 test examples. Each original email was processed using the processEmail and emailFeatures functions and converted into a vector x^(i) ∈ R^1899.

After loading the dataset, [ex6_spam.m](ex6_spam.m) will proceed to train a SVM to classify between spam (y = 1) and non-spam (y = 0) emails. Once the training completes, I see that the classifier gets a 

#### Training accuracy of about 99.8% and a Test accuracy of about 98.5%.

## Top Predictors for Spam

To better understand how the spam classifier works, I inspect the parameters to see which words the classifier thinks are the most predictive of spam. The next step of [ex6_spam.m](ex6_spam.m) finds the parameters with the largest positive values in the classifier and displays the corresponding words (Figure 12). Thus, if an email contains words such as “guarantee”, “remove”, “dollar”, and “price” (the top predictors shown in Figure 12), it is likely to be classified as spam.

Figure 12: Top predictors for spam email

![](https://github.com/raianilar17/Support-Vector-Machines/blob/master/ex6_spam_figure3.png)


## Try my own emails

Now that I have trained a spam classifier, I start trying it out on my own emails.I tried the list of examples and see if the classifier gets them right.

[emailSample1.txt](emailSample1.txt ) 

[emailSample2.txt](emailSample2.txt )

[spamSample1.txt](spamSample1.txt) 

[spamSample2.txt](spamSample1.txt)

[my_email.txt(plain text files)](my_email.txt)

[my_spam1.txt(plain text files)](my_spam1.txt)

[my_spam2.txt(plain text files)](my_spam1.txt)

## Prediction Table

|Test_Samples | Actual_Value | Prediction_Value|
|-------------|--------------|-----------------|
|emailSample1.txt|1|1|
|emailSample12.txt|1|1|
|spamSample1.txt|0|0|
|spamSample2.txt|0|0|
|my_email.txt|0|0|
|my_spam1.txt|1|1|
|my_spam2.txt|1|1|

### 100% Accuracy on unseen email(Test email).

The ouput of [ex6_spam.m](ex6_spam.m) octave script

Type on octave Cli 

ex6_spam

[output Looks like](output_ex6_spam.txt)


## Future work

Build  own dataset using the original emails from the [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/),then randomly divide up the dataset into a training set, a cross validation set and a test set and also build own vocabulary list (by selecting the high frequency words that occur in the dataset) and adding any additional features that think might be useful.

Finally, I will also try to use highly optimized SVM toolboxes such as [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [SVMLight](http://svmlight.joachims.org/).

Add more description

Apply this Algorithm in more real-world datasets.

## Work in progress....
