# Loads an index containing entity -> frequency pairs.


class EntityCountMap:
    def __init__(self, ent_counts_file, rltd=None):
        print('==> Loading entity counts map')
        self.ent_f_start = {}
        self.ent_f_end = {}
        cur_start = 1
        cnt = 0
        with open(ent_counts_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                parts = line.split('\t')
                ent_wikiid = int(parts[0])
                ent_f = int(parts[2])

                if not rltd or ent_wikiid not in rltd.wikiid_to_rltdid:
                    self.ent_f_start[ent_wikiid] = cur_start
                    self.ent_f_end[ent_wikiid] = cur_start + ent_f - 1
                    cur_start += ent_f
                    cnt += 1

        print('    Done loading entity freq index. Size = {}'.format(cnt))

    def get_ent_freq(self, ent_wikiid):
        if ent_wikiid in self.ent_f_start:
            return self.ent_f_end[ent_wikiid] - self.ent_f_start[ent_wikiid] + 1
        return 0
