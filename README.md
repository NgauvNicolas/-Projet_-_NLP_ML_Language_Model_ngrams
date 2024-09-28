# Projet_-_NLP_ML_Language_Model_ngrams
## Projet du cours de TAL Anglais du M1 (S1) consistant à apprécier un modèle de langue pour générer des avis à l'aide de n-grammes


NGAUV Nicolas
M1 PluriTAL INaLCO 

## PROJECT SEMESTER 1: ESTIMATING A LANGUAGE MODEL TO GENERATE REVIEWS




All the project have been done (entirely), please look the code and test to check it.

By default, when execute the file 'project.py', all the results are obtained using stupid backoff : to appreciate results without it, please follow the instructions below.

- Comment the line 460
- Uncomment the lines 431 to 447



When execute the code (python3 project.py), some results are print in the terminal (a generated review, its perplexity, ...) and some file are created (parameters file, file review, etc.)

To test all the answers to the exercices, you can comment and uncomment some test in the main() function.

I've commented the majority of the functions, I hope it's relatively clear...

With 'Prime_Pantry_test.txt' and 'Prime_Pantry_train.txt', we have similar perplexity values for a review generated and file of 100 reviews generated.
But I suggest you to only use 'Prime_Pantry_test.txt' when you test because with the other, it takes too long for the code to be fully executed and create very heavy file (more than 5 minutes for more than 200 Mio in total).
If you still want to test it, please:
- comment the lines 596 and 656
- uncomment the lines 597 and 657


When we are not using the stupid backoff, we do just like the question 2. c. : the test reviews are guaranteed to contain any ngram seen in the training corpus.
This is not the case when we use the stupid backoff : we work with the hypothesis of "what if a parameter still cannot be found during evaluation ?", so we can and consider that we can find when evaluating a parameter for a context of n which is not in the training vocab.
That's why the perplexity value is so different when using the stupid backoff or not (so low when not using stupid backoff and so high when using it).
