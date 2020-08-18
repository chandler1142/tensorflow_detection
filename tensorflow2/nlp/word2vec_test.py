from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('data/wiki_zh_jian_text.model')

testwords = ['国家电网']

for element in testwords:
    res = en_wiki_word2vec_model.most_similar(element)
    print(element)
    print(res)
