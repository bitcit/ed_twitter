# TODO: Use some arbitrary stop words.
#       (Only doing this because of the original code)

all_stop_words = {'a', 'about', 'above', 'across', 'after', 'afterwards',
                  'again', 'against', 'all', 'almost', 'alone', 'along',
                  'already', 'also', 'although', 'always', 'am', 'among',
                  'amongst', 'amoungst', 'amount', 'an', 'and', 'another',
                  'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere',
                  'are', 'around', 'as', 'at', 'back', 'be', 'became',
                  'because', 'become', 'becomes', 'becoming', 'been', 'before',
                  'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                  'between', 'beyond', 'both', 'bottom', 'but', 'by', 'call',
                  'can', 'cannot', 'cant', 'dont', 'co', 'con', 'could',
                  'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done',
                  'down', 'due', 'during', 'each', 'eg', 'eight', 'either',
                  'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc',
                  'even', 'ever', 'every', 'everyone', 'everything',
                  'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill',
                  'find', 'fire', 'first', 'five', 'for', 'former', 'formerly',
                  'forty', 'found', 'four', 'from', 'front', 'full', 'further',
                  'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he',
                  'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
                  'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                  'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
                  'interest', 'into', 'is', 'it', 'its', 'itself', 'keep',
                  'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made',
                  'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine',
                  'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must',
                  'my', 'myself', 'name', 'namely', 'neither', 'never',
                  'nevertheless', 'next', 'nine', 'no', 'nobody', 'none',
                  'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of',
                  'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or',
                  'other', 'others', 'otherwise', 'our', 'ours', 'ourselves',
                  'out', 'over', 'own', 'part', 'per', 'perhaps', 'please',
                  'put', 'rather', 're', 'same', 'see', 'seem', 'seemed',
                  'seeming', 'seems', 'serious', 'several', 'she', 'should',
                  'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so',
                  'some', 'somehow', 'someone', 'something', 'sometime',
                  'sometimes', 'somewhere', 'still', 'such', 'system', 'take',
                  'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
                  'then', 'thence', 'there', 'thereafter', 'thereby',
                  'therefore', 'therein', 'thereupon', 'these', 'they', 'thick',
                  'thin', 'third', 'this', 'those', 'though', 'three',
                  'through', 'throughout', 'thru', 'thus', 'to', 'together',
                  'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
                  'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via',
                  'was', 'we', 'well', 'were', 'what', 'whatever', 'when',
                  'whence', 'whenever', 'where', 'whereafter', 'whereas',
                  'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
                  'which', 'while', 'whither', 'who', 'whoever', 'whole',
                  'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
                  'would', 'yet', 'you', 'your', 'yours', 'yourself',
                  'yourselves', 'st', 'years', 'yourselves', 'new', 'used',
                  'known', 'year', 'later', 'including', 'used', 'end', 'did',
                  'just', 'best', 'using'}


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_stop_word_or_number(w):
    return w.lower() in all_stop_words or is_int(w) or len(w) < 2
