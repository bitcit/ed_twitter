# Loads the link redirect index from Wikipedia.


# Only used in ent_name_id.py
class WikiRedirectsIndex:
    def __init__(self, wiki_redirects_path):
        print('==> Loading redirects index')

        self.wiki_redirects_index = {}

        with open(wiki_redirects_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                self.wiki_redirects_index[parts[0]] = parts[1]
        print('    Done loading redirects index with size {}'.format(
            len(self.wiki_redirects_index)))

    def get_redirected_ent_title(self, ent_name):
        if ent_name in self.wiki_redirects_index:
            return self.wiki_redirects_index[ent_name]
        else:
            return ent_name
