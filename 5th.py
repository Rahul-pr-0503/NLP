training_data = [
    (['fun', 'couple', 'love', 'love'], 'comedy'),
    (['fast', 'furious', 'shoot'], 'action'),
    (['couple', 'fly', 'fast', 'fun', 'fun'], 'comedy'),
    (['furious', 'shoot', 'shoot', 'fun'], 'action'),
    (['fly', 'fast', 'shoot', 'love'], 'action')
]
new_document = ['fast', 'couple', 'shoot', 'fly']
total_docs = len(training_data)
total_comedy_docs = sum(1 for _, genre in training_data if genre == 'comedy')
total_action_docs = total_docs - total_comedy_docs
V = set(word for doc, _ in training_data for word in doc)
p_comedy = total_comedy_docs / total_docs
p_action = total_action_docs / total_docs
likelihood_comedy = 1.0
likelihood_action = 1.0
for word in new_document:
    count_word_comedy = sum(1 for doc, genre in training_data if genre == 'comedy' and word in doc)
    count_word_action = sum(1 for doc, genre in training_data if genre == 'action' and word in doc)
    p_word_comedy = (count_word_comedy + 1) / (total_comedy_docs + len(V))
    p_word_action = (count_word_action + 1) / (total_action_docs + len(V))
    likelihood_comedy *= p_word_comedy
    likelihood_action *= p_word_action
posterior_comedy = p_comedy * likelihood_comedy
posterior_action = p_action * likelihood_action
most_likely_class ='comedy' if posterior_comedy > posterior_action else 'action'
print('Posterior probability for comedy:', posterior_comedy)
print('Posterior probability for action:', posterior_action)
print('Most likely class for D:', most_likely_class)