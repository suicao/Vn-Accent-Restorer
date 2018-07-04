import codecs
import re
from tqdm import tqdm
from data_load import basic_tokenizer, load_vocab
import sys

raw_path = "./corpora/blog.txt"
processed_path = "./corpora/iblog_processed.txt"
vocab_path = "./corpora/blog_vocab.txt"
dictionary = dict()
vocab_size = 6400
add_unk = True

# result = []
# with codecs.open(raw_path, encoding="utf8") as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         if line[0] == "\n":
#             continue
#         line = line.replace(".",".\n").replace("?","?\n").replace("!","!\n")
#         result.append(line)
#
# with codecs.open(raw_path, "w", encoding="utf8") as f:
#     f.write(''.join(result))
# sys.exit()
if add_unk:
    with codecs.open(raw_path, encoding="utf8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line[0] == "\n":
                continue
            line += " <eos>"
            for word in basic_tokenizer(line):
                if word.isdigit():
                    word = "<num>"
                if word[0] != "<" and not word.isalpha():
                    continue
                if dictionary.get(word) is None:
                    dictionary[word] = 0
                dictionary[word] += 1

    dictionary["<unk>"] = 99999999
    count_pairs = sorted(dictionary.items(), key=lambda x: -x[1])
    tokens, _ = zip(*count_pairs)
    tokens = tokens[0:vocab_size]
    tokens = list(tokens)
    tokens.append("<pad>")
    print("Token count: {}".format(len(tokens)))
    print("Counting completed.")

    vocab = []
    for idx, token in enumerate(tokens):
        vocab.append("{} {}".format(token, idx))

    with codecs.open(vocab_path, "w", encoding="utf8") as f:
        f.write("\n".join(vocab))

    # result = []
    # with codecs.open(raw_path, encoding="utf8") as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         if line[0] == "\n":
    #             continue
    #         line += " <eos>"
    #         l = []
    #         for word in basic_tokenizer(line):
    #             if word.isdigit():
    #                 l.append("<num>")
    #             if word[0] != "<" and not word.isalpha():
    #                 continue
    #             if word in tokens:
    #                 l.append(word)
    #             else:
    #                 l.append("<unk>")
    #         result.append(' '.join(l))
    #
    # result = "\n".join(result)
    #
    # print("Writing to file.")
    #
    # with codecs.open(processed_path, "w", encoding="utf8") as f:
    #     f.write(result)

# train = []
# train_path = "corpora/train.txt"
# valid = []
# valid_path = "corpora/valid.txt"
# test = []
# test_path = "corpora/test.txt"
# with codecs.open(processed_path, "r", encoding="utf8") as f:
#     lines = f.readlines()
#     i = 0
#     for line in tqdm(lines):
#         if (i - 8) % 10 == 0:
#             valid.append(line)
#         elif (i - 9) % 10 == 0:
#             test.append(line)
#         else:
#             train.append(line)
#         i += 1
#
# with codecs.open(train_path, "w", encoding="utf8") as f:
#     f.write("".join(train))
# with codecs.open(valid_path, "w", encoding="utf8") as f:
#     f.write("".join(valid))
# with codecs.open(test_path, "w", encoding="utf8") as f:
#     f.write("".join(test))
