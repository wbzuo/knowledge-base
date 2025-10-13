vocab = ["apple", "banana", "cherry"]

word_to_index = {word: idx for idx, word in enumerate(vocab)}

word = word_to_index.keys()

def one_hot_encoding(word, vocab, word_to_index):
    encoding = [0] * len(vocab)
    
    # 获得idx
    idx = word_to_index.get(word, -1)
    if idx != -1:
        encoding[idx] = 1
    return encoding

# 测试
print("OneHot编码:")
for word in vocab:
    print(f"'{word}': {one_hot_encoding(word, vocab, word_to_index)}")