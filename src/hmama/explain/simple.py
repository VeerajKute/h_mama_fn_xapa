def explain_text_simple(text, model=None):
    # very simple explainability: show words and a mock importance score
    words = text.split()
    scores = [round(1.0/len(words), 3)]*len(words)
    return list(zip(words, scores))
