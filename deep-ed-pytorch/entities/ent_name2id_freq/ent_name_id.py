# ---------------- Load entity name-id mappings ------------------
#  Each entity has:
#    a) a Wikipedia URL referred as 'name' here
#    b) a Wikipedia ID referred as 'ent_wikiid' or 'wikiid' here
#    c) an ID that will be used in the entity embeddings lookup table.
#       Referred as 'ent_thid' or 'thid' here.

import os
import torch

from utils.utils import trim1, first_letter_to_uppercase
from data_gen.indexes.wiki_disambiguation_pages_index \
    import load_wiki_disambiguation_index
from data_gen.indexes.wiki_redirects_index import WikiRedirectsIndex
from entities.relatedness.relatedness import Relatedness


class EntityNameId:
    def __init__(self, args):
        self.wiki_redirects_index = WikiRedirectsIndex(
            os.path.join(args.root_data_dir, args.wiki_redirects))
        self.rltd_only = False
        self.rltd = None

        if 'entities' in args and args.entities != 'ALL':
            self.rltd = Relatedness(args)
            assert self.rltd.wikiid_to_rltdid
            self.rltd_only = True

        self.unk_ent_wikiid = 1
        entity_wiki_txtfilename = os.path.join(args.root_data_dir,
                                               args.wiki_name_id_map)

        ent_id_map_path = 'generated/ent_name_id_map_RLTD' \
            if self.rltd_only else 'generated/ent_name_id_map'
        entity_wiki_torch = os.path.join(args.root_data_dir,
                                         ent_id_map_path)

        print('==> Loading entity wikiid - name map')

        if os.path.exists(entity_wiki_torch):
            print('---> from torch file: ' + entity_wiki_torch)
            other = torch.load(entity_wiki_torch)
            self.ent_wikiid2name = other.ent_wikiid2name
            self.ent_name2wikiid = other.ent_name2wikiid
            if not self.rltd_only:
                self.ent_wikiid2thid = other.ent_wikiid2thid
                self.ent_thid2wikiid = other.ent_thid2wikiid

        else:
            print('---> torch file NOT found. Loading from disk (slower). '
                  'Out f = ' + entity_wiki_torch)
            wiki_disambiguation_index = load_wiki_disambiguation_index(
                os.path.join(args.root_data_dir, args.wiki_disambiguation))
            print('    Still loading entity wikiid - name map ...')

            # Map for entity name to entity wiki id
            self.ent_wikiid2name = {}
            self.ent_name2wikiid = {}

            # Map for entity wiki id to tensor id. Size = 4.4M
            if not self.rltd_only:
                self.ent_wikiid2thid = {}
                self.ent_thid2wikiid = {}

            cnt = 0
            with open(entity_wiki_txtfilename, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    parts = line.split('\t')
                    ent_name = parts[0]
                    ent_wikiid = int(parts[1])

                    wikiid_rltd = \
                        self.rltd and ent_wikiid in self.rltd.wikiid_to_rltdid

                    if ent_wikiid not in wiki_disambiguation_index:
                        if not self.rltd_only or wikiid_rltd:
                            self.ent_wikiid2name[ent_wikiid] = ent_name
                            self.ent_name2wikiid[ent_name] = ent_wikiid
                        if not self.rltd_only:
                            self.ent_wikiid2thid[ent_wikiid] = cnt
                            self.ent_thid2wikiid[cnt] = ent_wikiid
                            cnt += 1

            if not self.rltd_only:
                self.ent_wikiid2thid[self.unk_ent_wikiid] = cnt
                self.ent_thid2wikiid[cnt] = self.unk_ent_wikiid

            self.ent_wikiid2name[self.unk_ent_wikiid] = 'UNK_ENT'
            self.ent_name2wikiid['UNK_ENT'] = self.unk_ent_wikiid

            torch.save(self, entity_wiki_torch)

        if self.rltd_only:
            self.unk_ent_thid = self.rltd.wikiid_to_rltdid[self.unk_ent_wikiid]
            total_num_ents = len(self.rltd.wikiid_to_rltdid)
        else:
            self.unk_ent_thid = self.ent_wikiid2thid[self.unk_ent_wikiid]
            total_num_ents = len(self.ent_wikiid2thid)

        print('    Done loading entity name - wikiid. Size thid index = {}'
              .format(total_num_ents))

    def preprocess_ent_name(self, ent_name):
        ent_name = trim1(ent_name)
        ent_name = ent_name.replace('&amp;', '&')
        ent_name = ent_name.replace('&quot;', '"')
        ent_name = ent_name.replace('_', ' ')
        ent_name = first_letter_to_uppercase(ent_name)

        if self.wiki_redirects_index:
            ent_name = self.wiki_redirects_index.get_redirected_ent_title(
                ent_name)
        return ent_name

    def get_ent_name_from_wikiid(self, ent_wikiid):
        if ent_wikiid not in self.ent_wikiid2name:
            return

        return self.ent_wikiid2name[ent_wikiid]

    def get_ent_wikiid_from_name(self, ent_name, not_verbose=False):
        ent_name = self.preprocess_ent_name(ent_name)
        if not ent_name or ent_name not in self.ent_name2wikiid:
            if not not_verbose:
                print('Entity {} not found. Redirects file needs to be loaded '
                      'for better performance.'.format(ent_name))
            return self.unk_ent_wikiid
        return self.ent_name2wikiid[ent_name]

    def ent_from_wikiid(self, ent_wikiid):
        if not ent_wikiid or ent_wikiid not in self.ent_wikiid2name:
            return 'NIL'
        return self.ent_wikiid2name[ent_wikiid]

    def get_wikiid_from_thid(self, ent_thid):
        if not ent_thid:
            return self.unk_ent_wikiid
        if self.rltd_only:
            ent_wikiid = self.rltd.rltdid_to_wikiid[ent_thid]
        else:
            ent_wikiid = self.ent_thid2wikiid[ent_thid]
        if not ent_wikiid:
            return self.unk_ent_wikiid
        return ent_wikiid

    def get_map_all_valid_ents(self):
        m = {}
        for ent_wikiid in self.ent_wikiid2name:
            m[ent_wikiid] = 1
        return m

    def is_valid_ent(self, ent_wikiid):
        return ent_wikiid in self.ent_wikiid2name

    def get_total_num_ents(self):
        if self.rltd_only:
            return len(self.rltd.wikiid_to_rltdid)
        else:
            return len(self.ent_thid2wikiid)

    # ent wiki id -> thid
    def get_thid(self, ent_wikiid):
        if not ent_wikiid:
            return self.unk_ent_thid
        if self.rltd_only:
            if ent_wikiid not in self.rltd.wikiid_to_rltdid:
                return self.unk_ent_thid
            return self.rltd.wikiid_to_rltdid[ent_wikiid]
        else:
            if ent_wikiid not in self.ent_wikiid2thid:
                return self.unk_ent_thid
            return self.ent_wikiid2thid[ent_wikiid]

    # tensor of ent wiki ids --> tensor of thids
    def get_ent_thids(self, ent_wikiids_tensor):
        ent_thid_tensor = ent_wikiids_tensor.clone()
        if ent_wikiids_tensor.dim() == 2:
            for i in range(ent_thid_tensor.size(0)):
                for j in range(ent_thid_tensor.size(1)):
                    ent_thid_tensor[i][j] = self.get_thid(
                        int(ent_wikiids_tensor[i][j]))
        elif ent_wikiids_tensor.dim() == 1:
            for i in range(ent_thid_tensor.size(0)):
                ent_thid_tensor[i] = self.get_thid(int(ent_wikiids_tensor[i]))
        else:
            print('Tensor with > 2 dimentions not supported')
            os.exit()
        return ent_thid_tensor
