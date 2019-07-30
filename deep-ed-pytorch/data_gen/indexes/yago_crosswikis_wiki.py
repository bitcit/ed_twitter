# Loads the merged p(e|m) index.

from utils.utils import modify_uppercase_phrase


class YagoCrosswikisIndex:
    def __init__(self, crosswikis_path, yago_p_e_m_path):
        self.ent_p_e_m_index = {}
        self.mention_lower_to_one_upper = {}
        self.mention_total_freq = {}

        print('==> Loading crosswikis_wikipedia from file ' + crosswikis_path)
        num_lines = 0

        with open(crosswikis_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                num_lines += 1
                if num_lines % 2000000 == 0:
                    print('Processed {} lines.'.format(num_lines))

                parts = line.split('\t')
                mention = parts[0]
                total = int(parts[1])
                if total >= 1:
                    self.ent_p_e_m_index[mention] = {}
                    self.mention_lower_to_one_upper[mention.lower()] = mention
                    self.mention_total_freq[mention] = total
                    num_parts = len(parts)
                    for i in range(2, num_parts):
                        ent_str = parts[i].split(',')
                        ent_wikiid = int(ent_str[0])
                        freq = int(ent_str[1])
                        # not sorted
                        self.ent_p_e_m_index[mention][ent_wikiid] = freq / total

        print('==> Loading yago index from file {}'.format(yago_p_e_m_path))
        num_lines = 0

        with open(yago_p_e_m_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                num_lines += 1
                if num_lines % 2000000 == 0:
                    print('Processed {} lines.'.format(num_lines))

                parts = line.split('\t')
                mention = parts[0]
                total = int(parts[1])
                if total < 1:
                    continue
                self.mention_lower_to_one_upper[mention.lower()] = mention
                if mention not in self.mention_total_freq:
                    self.mention_total_freq[mention] = 0
                self.mention_total_freq[mention] += total

                yago_ment_ent_idx = {}

                num_parts = len(parts)
                for i in range(2, num_parts):
                    ent_str = parts[i].split(',')
                    ent_wikiid = int(ent_str[0])
                    freq = 1
                    # not sorted
                    yago_ment_ent_idx[ent_wikiid] = freq / total

                if mention not in self.ent_p_e_m_index:
                    self.ent_p_e_m_index[mention] = yago_ment_ent_idx
                else:
                    for ent_wikiid in yago_ment_ent_idx:
                        if ent_wikiid not in self.ent_p_e_m_index[mention]:
                            self.ent_p_e_m_index[mention][ent_wikiid] = 0.0

                        self.ent_p_e_m_index[mention][ent_wikiid] = \
                            min(1.0, self.ent_p_e_m_index[mention][ent_wikiid] +
                                yago_ment_ent_idx[ent_wikiid])

        print('    Done loading index')

    # Function used to preprocess a given mention such that it has higher
    # chance to have at least one valid entry in the p(e|m) index.
    def preprocess_mention(self, m):
        cur_m = modify_uppercase_phrase(m)
        if cur_m not in self.ent_p_e_m_index:
            cur_m = m
        if m in self.mention_total_freq and self.mention_total_freq[m] > \
                self.mention_total_freq[cur_m]:
            # Cases like 'U.S.' are handed badly by modify_uppercase_phrase
            cur_m = m
        # If we cannot find the exact mention in our index, we try our luck to
        # find it in a case insensitive index.
        if cur_m not in self.ent_p_e_m_index and \
                cur_m.lower() in self.mention_lower_to_one_upper:
            cur_m = self.mention_lower_to_one_upper[cur_m.lower()]
        return cur_m
