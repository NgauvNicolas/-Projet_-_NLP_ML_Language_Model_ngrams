#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created By  : NGAUV Nicolas (M1 PluriTAL (INaLCO) 22301604)




from lib2to3.pgen2.literals import test
import re
import pickle
from collections import defaultdict
import numpy as np
import random
import math
import collections





'''
1. Estimating the Language Model

    a. 
The probability of observing the word 𝑤𝑖 given the two previous words 𝑤𝑖−1 and 𝑤𝑖−2 in an n-gram language model is estimated using conditional probability. Mathematically, it is expressed as:
P(wi∣wi−1,wi−2,…,wi−n+1)

This conditional probability is estimated by calculating the relative frequency of the trigram (in this example) (wi−2,wi−1,wi) in the training corpus. The formula for estimating the probability is:
P(wi∣wi−1,wi−2) = (Count(wi−2,wi−1,wi))​ / (Count(wi−2,wi−1))

Where:
    Count(wi−2,wi−1,wi) is the number of occurrences of the trigram (wi−2,wi−1,wi) in the training corpus.
    Count(wi−2,wi−1) is the number of occurrences of the bigram (wi−2,wi−1) in the training corpus.

This approach generalizes to higher-order n-gram models as well. In the case of trigrams, we are considering the two previous words as context to estimate the probability of the current word.

In summary, the probability of observing a word at position wi in a sentence is estimated based on the relative frequency of the trigram (wi−2,wi−1,wi) in the training data.


    c.
Adding special tokens such as <s> (start token) and </s> (end token) to the beginning and end of each review is a common practice in natural language processing and language modeling. This technique is part of a process known as "padding" or "sequence padding." Here's why it's useful:
    Contextual Information:
        The start token <s> added n-1 times at the beginning of the review helps the language model to establish context. It indicates the model that it is at the beginning of a sequence and sets the initial context for predicting the next word.
    Modeling Sequences:
        Language models, especially those based on n-grams, rely on the context of previous words to predict the next word. By adding start tokens, we provide the model with a context that spans the initial n-1 positions, which is crucial for accurate predictions.
    Consistent Input Length:
        Padding ensures that all input sequences have a consistent length. This is important for the efficient processing of data, especially when working with batches. Most machine learning frameworks expect input sequences to have the same length, and padding helps achieve this uniformity.
    Handling Variable-Length Sequences:
        In natural language processing, text data often comes in variable-length sequences. Padding ensures that all sequences have a uniform length, making it easier to work with them in a computational context.
    Marking End of Sequence:
        The end token </s> signals the end of the sequence. This is useful for language models to understand where the input sequence concludes, allowing the model to capture dependencies within the given context.
    Improved Training Stability:
        Adding start tokens at the beginning of the sequence helps the model initialize its internal state, potentially improving the stability of training.

In summary, by adding start and end tokens, we provide the language model with necessary contextual information, make input sequences consistent in length, and help the model better understand the structure and boundaries of the input data. This preprocessing step contributes to the overall performance and effectiveness of the language model.





2. Generation

    a. 
To generate new sentences based on the conditional probability distribution for a particular n-1-gram, we need to initialize the history appropriately. The history represents the context used to predict the next word. In the case of trigrams (n=3), the history consists of the two previous words (𝑤𝑖−2, 𝑤𝑖−1).
    For the First n-1 Words:
        For the first (n-1) words of the sentence, we can use special tokens like <s> (start of sentence) or a repetition of a padding token to fill in the initial history.
    For Subsequent Words:
        As we generate each new word, update the history by shifting the existing history and adding the newly generated word. In the case of trigrams, update the history as follows: h <-- wi−1, wi.

        

    b.
The punctuation is considerated as token (except '-'), just like words, so there are space before and after each of them and it's a little bit unnatural.
In order to correct that, we use : 
review = re.sub("( [,.;:!?\)\]]|[\[\(~$£€] | ['&+/] )", lambda m: m.group(1).strip(), " ".join(generated_sentence))
It allow us to considerate spaces or not with left and right contexts for ponctuation and symbols (but not for "" and smileys like :( ))


Observations when using different n-gram models (1-gram, 2-gram, 3-gram, and 4-gram) in language modeling:

    1-gram Model:
        Simplest model where each word is considered independent of others.
        Generated sentences may lack coherence and context.
        Limited understanding of word relationships and dependencies.

    2-gram Model:
        Considers the probability of a word based on the previous word.
        Improved coherence compared to 1-gram, but still simplistic.
        Captures some local context but may miss longer-range dependencies.

    3-gram Model:
        Takes into account the probability of a word given the two previous words.
        Better captures local context and dependencies.
        Sentences may exhibit improved coherence and exhibit some understanding of phrase structures.

    4-gram Model:
        Considers the probability of a word based on the three previous words.
        Further improvement in capturing context and dependencies.
        Sentences are likely to exhibit more coherent structures and better reflect the training data's patterns.

    General Observations:
        As the n-gram order increases, the model captures more extensive contextual information, leading to more coherent and contextually relevant sentences.
        Higher-order n-gram models are more sensitive to the specifics of the training data and may generate sentences that closely resemble the training set.
        However, higher-order n-gram models also face challenges such as data sparsity, especially when dealing with long contexts.

    Common Limitations:
        All n-gram models may struggle with handling unseen combinations of words, resulting in frequent use of backoff strategies like '<unk>' token replacement or 'stupid' backoff.
        The generated sentences may still lack true semantic understanding and creativity, as n-gram models are inherently limited by the fixed context size.

    Training Data Impact:
        The quality and diversity of the training data significantly influence the model's ability to generalize and generate coherent sentences.

The effectiveness of each n-gram model depends on the specific characteristics of our data and the complexity of the language patterns it contains. It's common to experiment with different orders of n-gram models and choose the one that strikes the right balance between capturing context and avoiding data sparsity issues.  
        
        
'''








def make_ngrams(sentence, n):
    # Séparate ponctuations from words
    words = re.findall(r'[\w-]+|<unk>|[^\w\s]', sentence)


    # Check value n
    if n <= 0 or n > len(words):
        raise ValueError("Invalide value n.")
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams




def preprocess_review(review, n):
    # Separate ponctuations from words
    words = re.findall(r'[\w-]+|<unk>|[^\w\s]', review)

    # Add start tokens <s> n-1 times
    start_tokens = ['<s>'] * (n - 1)
    
    # Add end token </s>
    #end_token = ['</s>']
    end_token = ['</s>'] * (n - 1)

    # Combine tokens and review
    preprocessed_review = start_tokens + words + end_token
    #preprocessed_review = start_tokens + make_ngrams(review, n) + end_token
    
    return preprocessed_review








def make_conditional_probas(file_path, n):
    # Initialize count table
    count_table = defaultdict(lambda: defaultdict(int))

    # Read the file and process reviews
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Preprocess the review by adding start and end tokens
            review = preprocess_review(line.strip(), n)

            # Update count table with n-grams
            for i in range(len(review) - n + 1):
                ngram = tuple(review[i:i+n])
                next_word = review[i + n] if i + n < len(review) else 'END'
                count_table[ngram][next_word] += 1

    # Calculate conditional probabilities
    conditional_probas = defaultdict(dict)
    for ngram, next_words in count_table.items():
        total_count = sum(next_words.values())
        for next_word, count in next_words.items():
            conditional_probas[ngram][next_word] = count / total_count

    return conditional_probas







def create_parameters(file_path, max_ngram):
    parameters = {}

    for n in range(1, max_ngram + 1):
        conditional_probas = make_conditional_probas(file_path, n)
        parameters[f'{n}-gram'] = conditional_probas

    return parameters



def serialize_parameters(parameters, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(parameters, file)



def load_parameters(file_path):
    with open(file_path, 'rb') as file:
        parameters = pickle.load(file)
    return parameters







def sample_from_discrete_distrib(distrib):
    words, probas = zip(*distrib.items())
    probas = np.asarray(probas).astype('float64')/np.sum(probas)
    return np.random.choice(words, p=probas)







def initialize_history(n):
    # Initialize history with special tokens or padding
    #history = ['<s>'] * (n - 1)
    history = ['<s>'] * n
    return history

def update_history(history, new_word):
    # Update history by shifting and adding the new word
    history = history[1:] + [new_word]
    return history






# Generate sentences with a probability table and a number n (for n-grams model chosen) between 1 and 4
def generate(probability_table, n):
    # Initialize history for the first n-1 words
    #n = len(list(probability_table.keys())[0]) - 1
    #print(probability_table.keys())
    #print(list(probability_table.keys())[0])

    #print(list(probability_table.values())[n-1].keys())
    #print(list(probability_table.values())[n-1].values())
    #print(list(probability_table.values())[n-1].values())
    #print(list(probability_table.values())[n-1])
    
    l = list(probability_table.values())[n-1]
    #print(l)


    #print(list(probability_table.values())[0].keys())
    #print(list(probability_table.values())[3].keys()[0])
    #print(probability_table.values())
    #print((list(probability_table.values())[0].keys())[0])

    history = initialize_history(n)
    #print(history)
    history = update_history(history, "Rating")
    #print(history)

    #print(n)

    # Generate sentences until </s> token is encountered
    generated_sentence = []
    generated_sentence.append("Rating")
    while True:
        # Get the conditional probabilities for the current history
        current_history = tuple(history)
        #current_history = random.choices(tuple(probability_table.values()))[0]
        #current_history = random.choices(tuple(probability_table.values()))
        #print(current_history)
        #print("yeah")
        # n-1 to reach n-1 grams
        if current_history not in (l.keys()):
            #print("Noooooo")
            break  # Exit if no more predictions available for the current history
        #print("Yeeeeees")
        #probabilities = probability_table[current_history]
        # Dictionaries list of probabilities
        #liste_prob = []
        # Dictionary of probabilities
        dico_prob = {}
        for x in l:
            #print(x)
            #print(x[0])
            if (x == current_history):
                #print("Ok")
                p = l[x]
                #print(l[x])
                #print(p)
                #list_prob.append(p)
                dico_prob.update(p)
                #print("OH")
                #print(liste_prob)
                #print(dico_prob)


        # Choose the next word based on probabilities
        #next_word = sample_from_discrete_distrib(probabilities)
        #next_word = random.choices(list(probabilities.keys()), weights=probabilities.values())[0]
        list_next_word = random.choices(list(dico_prob.keys()), weights=dico_prob.values())
        #print(list_next_word)
        next_word = list_next_word[0]
        #print(next_word)

        # Update history and add the word to the generated sentence
        history = update_history(history, next_word)
        generated_sentence.append(next_word)

        # Exit if </s> token is generated
        if next_word == '</s>':
            # Don't take the token in the review
            generated_sentence.remove(next_word)
            break
    # To considerate spaces or not with left context and right for ponctuation and symbols (but not for "" and smileys like :( ))
    review = re.sub("( [,.;:!?\)\]]|[\[\(~$£€] | ['&+/] )", lambda m: m.group(1).strip(), " ".join(generated_sentence))
    #print(review)
    return review




# For a file with n reviews generated
def generate_file(file, probability_table, n_gram, n):
    i = 0
    with open(file, 'w+', encoding='utf-8') as f:
        while i < n:
            review = generate(probability_table, n_gram)
            f.write(review)
            f.write("\n")
            i = i + 1








# 3. STUPID BACKOFF
# Code a recursive function over smaller and smaller context to implement the stupid backoff 
    # l is a probability table in list of n-grams format, 
    # current_history is the context in n-gram format, 
    # word is the word we need to find it's probability given the context
    # words is the list of all the words in the review(s)
def get_prob(l, current_history, word, words):
    #print(word)
    #print(current_history)

    
    list_prob = []
    for x in l:
        #print(x)
        #print(x[0])
        
        if (x == current_history):
            #print("Ok")
            #print(l[x])
            #print(p)
            p = l[x]
            list_prob.append(p)
            #print(list((list_prob[0]).values())[0])
            return list((list_prob[0]).values())[0]
            #proba_word = list((list_prob[0]).values())[0]
            #return proba_word
            #return l[x]
        
        elif (len(current_history) < 2):
            #print(len(words))
            return 1 / (len(words))
            #return math.exp(-100)
            

        else:
            #print("NOOOOO")
            # Convert tuple into list to modify
            current_history_list = list(current_history)
            #print(current_history_list)
            modif_current_history_list = current_history_list[1:]
            #print(modif_current_history_list)
            current_history_modif = tuple(modif_current_history_list)
            #print(current_history_modif)
            #print(0.4*get_prob(l, current_history_modif, word, words))
            return 0.4*get_prob(l, current_history_modif, word, words)
        








def calculate_perplexity(review, probability_table, n):
    l = list(probability_table.values())[n-1]
    history = initialize_history(n)
    history = update_history(history, "Rating")
    #perplexity_sum = 0
    #perplexity_sum = -math.log2(1)
    perplexity_sum = -math.log(1)
    inverse_probability_product = 1.0  # Initial value for the product
    #perp = 0
    #word_count = 0
    word_count = 1

    words = re.findall(r'[\w-]+|<unk>|[^\w\s]', review)

    for word in words[1:]:
        #print(word)
        current_history = tuple(history)
        #print(current_history)



        '''
        if current_history not in (l.keys()):
            continue  # Skip if no probabilities available for the current history
        list_prob = []
        for x in l:
            #print(x)
            #print(x[0])
            if (x == current_history):
                #print("Ok")
                p = l[x]
                #print(l[x])
                #print(p)
                list_prob.append(p)
                #dico_prob.update(p)

        word_probability = list((list_prob[0]).values())[0]
        '''

        #p = get_prob(l, current_history, word)

        #word_probability = probabilities.get(word, 1e-10)  # Use a small default probability for unseen words
        #word_probability = list_prob[0]
        #word_probability = list((list_prob[0]).values())[0]


        '''
        FOR 3. STUPID BACKOFF : 
        TO NOT TEST IT AND HAVING THE RESULT WITHOUT IT, COMMENT THE LINE BELOW AND UNCOMMENT THE INSTRUCTIONS BETWEEN ''' ''' JUST ABOVE
        '''
        word_probability = get_prob(l, current_history, word, words)


        #print(word_probability)
        #perplexity_sum += -math.log2(word_probability)
        perplexity_sum += -math.log(word_probability)
        inverse_probability_product *= 1 / word_probability
        #perp += -math.log(get_prob(l, , words[word]))
        word_count += 1

        history = update_history(history, word)
        #print(history)

    if word_count == 0:
        return float('inf')  # Return infinity if no words in the review

    #perplexity = 2 ** (perplexity_sum / word_count)
    #perplexity = math.exp((perplexity_sum / word_count))
    perplexity = inverse_probability_product ** (1 / word_count)
    #print(perplexity_sum)
    #print(word_count)
    #print(f"Perplexity1: {perplexity1}")
    #print(f"Perplexity: {perplexity}")

    return perplexity
    #return perplexity1
    



# The average perplexity over all sentences in the file reviews.
def calculate_average_perplexity(test_file_path, probability_table, n):
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_reviews = file.readlines()

    total_perplexity = 0
    total_reviews = 0

    for review in test_reviews:
        perplexity = calculate_perplexity(review, probability_table, n)
        
        if perplexity != float('inf'):  # Skip sentences with no words
            total_perplexity += perplexity
            total_reviews += 1

    average_perplexity = total_perplexity / total_reviews if total_reviews > 0 else float('inf')
    return average_perplexity









def replace_low_frequency_tokens(training_reviews, frequency_threshold=1):
    # Flatten the list of training reviews into a single list of words
    all_words = []
    with open(training_reviews, 'r', encoding='utf-8') as file, open(f"{training_reviews}_unk.txt", 'w+', encoding='utf-8') as unk:
        f = file.readlines()
        for l in f:
            words = re.findall(r'[\w-]+|<unk>|[^\w\s]', l)
            all_words.extend(words)

        # Count the frequency of each word
        word_frequency = collections.Counter(all_words)

        # Copy the training_reviews in "f{training_reviews}_unk.txt", then identify tokens with frequency less than the threshold and replace them with '<unk>'
        for l in f:
            list_words = []
            words = re.findall(r'[\w-]+|<unk>|[^\w\s]', l)
            for w in words:
                if word_frequency[w] <= frequency_threshold:
                    list_words.append('<unk>')
                else:
                    list_words.append(w)
            # To considerate spaces or not with left context and right for ponctuation and symbols (but not for "" and smileys like :( ))
            review = re.sub("( [,.;:!?\)\]]|[\[\(~$£€] | ['&+/] )", lambda m: m.group(1).strip(), " ".join(list_words))
            unk.write(review)
            unk.write('\n')


def replace_unseen_tokens_with_unk(test_review, training_reviews_unk):
    b = False
    # Copy test_review in "f{test_review}_unk.file and replace tokens not present in the training vocabulary (training_reviews_unk) with '<unk>'
    with open(training_reviews_unk, 'r', encoding='utf-8') as training_file, open(f"{test_review}_unk.txt", 'w+', encoding='utf-8') as test_file_unk, open(test_review, 'r', encoding='utf-8') as test_file:
        tf = test_file.readlines()
        training_unk = training_file.readlines()
        for x in tf:
            list_words = []
            words_x = re.findall(r'[\w-]+|<unk>|[^\w\s]', x)
            for wx in words_x:
                for y in training_unk:
                    words_y = re.findall(r'[\w-]+|<unk>|[^\w\s]', y)
                    if wx in words_y:
                        list_words.append(wx)
                        b = True
                        break
                if b == False:
                    list_words.append('<unk>')
            # To considerate spaces or not with left context and right for ponctuation and symbols (but not for "" and smileys like :( ))
            review = re.sub("( [,.;:!?\)\]]|[\[\(~$£€] | ['&+/] )", lambda m: m.group(1).strip(), " ".join(list_words))
            test_file_unk.write(review)
            test_file_unk.write('\n')
                
















def main():

    # 1.b. and 1.c.
    # Example usage:
    #sentence = "I love chocolate ice-cream."
    # n-grams' length
    #n = 3
    #result = make_ngrams(sentence, n)
    #print(result)
    #preprocessed_result = preprocess_review(sentence, n)
    #print(preprocessed_result)

    # 1.d. but we need this for the others questions
    n = 4
    file_path = 'Prime_Pantry_test.txt'  
    #file_path = 'Prime_Pantry_train.txt'
    conditional_probas_result = make_conditional_probas(file_path, n)
    #print(conditional_probas_result)
    # For 1-gram, 2-gram, 3-gram, 4-gram language models
    max_ngram = 4
    parameters = create_parameters(file_path, max_ngram)
    serialize_parameters(parameters, 'parameters.pkl')  # Serialize parameters
    loaded_parameters = load_parameters('parameters.pkl')  # Load parameters
    #print(loaded_parameters)
    print("Parameters loaded \n")


    # 2.a.
    # Example usage for trigrams (n=3)
    #n = 3
    #initial_history = initialize_history(n)
    #print("Initial History:", initial_history)
    # Update history for the next word
    #new_word = "example"
    #updated_history = update_history(initial_history, new_word)
    #print("Updated History:", updated_history)


    # 2.b. and 2.c.
    # Example usage:
    # Generate a review, and calculate its perplexity
    #n = 4
    generated_review = generate(loaded_parameters, n)
    print("A review generated and its perplexity :")
    print(generated_review)
    perplexity_review = calculate_perplexity(generated_review, loaded_parameters, n)
    print(perplexity_review)
    #print("PP: ", perplexity_review)
    print("\n")

    # Example usage:
    # Assuming file_path contains one review per line
    #average_perplexity = calculate_average_perplexity(file_path, loaded_parameters, n)
    #print(average_perplexity)

    # Exemple usage:
    # A file with 100 reviews is generated, and we calculate the perplexity of 
    generate_file("my_reviews.txt",loaded_parameters, n, 100)
    print("A file of 100 reviews generated and the average perplexity (look the file in the directory):")
    average_perplexity = calculate_average_perplexity("my_reviews.txt", loaded_parameters, n)
    print(average_perplexity)
    print("\n")
    print("\n")
    print("\n")


    # 3. Use of the <unk> token and stupid backoff
    # Example usage:
    # Copy the training_reviews in "{file_path}_unk.txt" (here: 'Prime_Pantry_test.txt_unk.txt'), then identify tokens with frequency less than the threshold and replace them with '<unk>'
    print("WITH THE <UNK> TOKENS NOW :")
    replace_low_frequency_tokens(file_path)

    # Now, use the file "{file_path}_unk.txt" (here: "Prime_Pantry_test.txt_unk.txt") as the new file_path and then computing the parameters as before
    #for 1-gram, 2-gram, 3-gram, 4-gram language models
    file_path_unk = 'Prime_Pantry_test.txt_unk.txt'
    #file_path_unk = 'Prime_Pantry_train.txt_unk.txt'
    max_ngram = 4
    parameters_unk = create_parameters(file_path_unk, max_ngram)
    serialize_parameters(parameters_unk, 'parameters_unk.pkl')  # Serialize parameters
    loaded_parameters_unk = load_parameters('parameters_unk.pkl')  # Load parameters
    print("Parameters with <unk> tokens loaded \n")

    # Generate a review, and calculate its perplexity
    generated_review_unk = generate(loaded_parameters_unk, n)
    print("A review generated and its perplexity :")
    print(generated_review_unk)
    perplexity_review_unk = calculate_perplexity(generated_review_unk, loaded_parameters_unk, n)
    print(perplexity_review_unk)
    print("\n")

    # A file with 100 reviews is generated, and we calculate the perplexity of 
    generate_file("my_reviews.txt",loaded_parameters_unk, n, 100)
    print("A file of 100 reviews generated and the average perplexity (look the file in the directory):")
    replace_unseen_tokens_with_unk("my_reviews.txt", file_path_unk)
    average_perplexity_unk = calculate_average_perplexity("my_reviews.txt_unk.txt", loaded_parameters_unk, n)
    print(average_perplexity_unk)
    print("\n")
    print("ALL THE RESULTS HERE ARE OBTAINED USING STUPID BACKOFF : TO APPRECIATE RESULTS WITHOUT IT, LOOK THE README.TXT")









if __name__ == "__main__":
    main()





