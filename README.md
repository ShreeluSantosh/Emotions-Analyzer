# Emotions-Analyzer
Currently in development

Credits: [Google Research - GoEmotions](https://github.com/google-research/google-research/tree/3c7c1ddae388ae79a4c27761b71c187d0e3b5f5e/goemotions)

I've used the first dataset from GoEmotions research for training my model.

This Emotions-Analyzer can classify the emotions detected in a given text as positive or negative.

Out of the 27 emotion labels used in the original research, I've classified them into 3 categories:
<ul>
<li><b><u>Positive Emotions</u></b>: Amusement, Approval, Admiration, Caring, Curiosity, Desire, Excitement, Gratitude, Joy, Love, Optimism, Pride, Relief, 
<li><b><u>Negtaive Emotions</u></b>: Anger, Annoyance, Disappointment, Disapproval, Disgust, Embarrassment, Fear, Grief, Nervousness, Remorse, Sadness
<li><b><u>Neutral</u></b> (not included in the classification): Realization, Surprise, Neutral
</ul>

So far, the accuracy of my model for positive emotions is 61.9%  and for negative emotions, it is 43.32%
