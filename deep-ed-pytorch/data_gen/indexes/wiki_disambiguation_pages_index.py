# Loads the link disambiguation index from Wikipedia


# TODO: only used in ent_name_id.py
def load_wiki_disambiguation_index(wiki_disambiguation_path):
    print('==> Loading disambiguation index')

    wiki_disambiguation_index = {}

    with open(wiki_disambiguation_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split("\t")
            assert int(parts[0])
            wiki_disambiguation_index[int(parts[0])] = 1

    print('    Done loading disambiguation index')
    print('    Size: {}'.format(len(wiki_disambiguation_index)))
    return wiki_disambiguation_index
