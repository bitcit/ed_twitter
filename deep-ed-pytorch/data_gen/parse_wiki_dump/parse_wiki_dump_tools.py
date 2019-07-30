# Utility functions to extract the text and hyperlinks from each page in the
# Wikipedia corpus.


# Extracts text and hyperlinks from each line of wikipedia.
def extract_text_and_hyp(line, mark_mentions, entity_name_id):
    list_hyp = []  # (mention, entity, count) tuples
    text = ''

    list_ent_errors = 0
    parsing_errors = 0
    disambiguation_ent_errors = 0
    diez_ent_errors = 0

    # TODO: make constants for all constant strings.
    begin_start_hyp = line.find('<a href="')
    end_start_hyp = begin_start_hyp + len('<a href="')

    end_end_hyp = 0
    begin_end_hyp = -1

    num_mentions = 0

    # An anchor has been found. (Otherwise begin_start_hyp == -1)
    while begin_start_hyp >= 0:
        # Extract text since the last hyperlink until this one.
        text += line[end_end_hyp:begin_start_hyp]

        next_quotes = line.find('">', end_start_hyp)
        end_quotes = next_quotes + len('">')
        if next_quotes < 0:
            parsing_errors += 1
            break

        begin_end_hyp = line.find('</a>', end_quotes)
        end_end_hyp = begin_end_hyp + len('</a>')
        if begin_end_hyp < 0:
            parsing_errors += 1
            break

        # Name of the entity.
        ent_name = line[end_start_hyp:next_quotes]

        # Text of the mention.
        mention = line[end_quotes:begin_end_hyp]
        mention_marker = False

        if (len(mention) > 0 and 'Wikipedia' not in mention
                and 'wikipedia' not in mention):
            # Remove prefix if present.
            ent_name = ent_name[5:] if ent_name.find('wikt:') == 0 else ent_name

            # Replace special characters and check redirects index.
            ent_name = entity_name_id.preprocess_ent_name(ent_name)

            i = ent_name.find('List of ')
            if i != 0:
                if '#' in ent_name:
                    diez_ent_errors += 1
                else:
                    # Get wikiid from name-entity map.
                    ent_wikiid = entity_name_id.get_ent_wikiid_from_name(
                        ent_name, True)
                    if ent_wikiid == entity_name_id.unk_ent_wikiid:
                        disambiguation_ent_errors += 1
                    else:
                        # Valid (entity,mention) pair.
                        num_mentions += 1
                        list_hyp.append({'mention': mention,
                                         'ent_wikiid': ent_wikiid,
                                         'cnt': num_mentions})
                        mention_marker = mark_mentions
            else:
                list_ent_errors += 1

        if not mention_marker:
            text += ' {} '.format(mention)
        else:
            text += ' MMSTART{} {} MMEND{} '.format(num_mentions, mention,
                                                    num_mentions)

        begin_start_hyp = line.find('<a href="', end_start_hyp)
        end_start_hyp = begin_start_hyp + len('<a href="')

    # Append the rest of the text.
    if begin_end_hyp >= 0 and end_end_hyp < len(line):
        text += line[end_end_hyp:]
    else:
        if not mark_mentions:
            text = line  # Don't throw away this line despite bad parsing.
        else:
            text = ''
            list_hyp = []

    return (
        list_hyp, text, list_ent_errors, parsing_errors,
        disambiguation_ent_errors, diez_ent_errors)

# TODO: Tests
# # ---------------------------- Unit tests -------------
# print('Unit tests:')
# test_line_1 = '<a href="Anarchism">Anarchism</a> is a <a href="political ' \
#               'philosophy">political philosophy</a> that advocates<a href=' \
#               '"stateless society">stateless societies</a>often defined as ' \
#               '<a href="self-governance">self-governed</a> voluntary ' \
#               'institutions, but that several authors have defined as more ' \
#               'specific institutions based on non-<a href="Hierarchy">' \
#               'hierarchical</a> <a href="Free association (communism and ' \
#               'anarchism)">free associations</a>..<a href="Anarchism">' \
#               'Anarchism</a>'
# test_line_2 = 'CSF pressure, as measured by <a href="lumbar puncture">lumbar ' \
#               'puncture</a> (LP), is 10-18 ' \
#               '<a href="Pressure#H2O">'
# test_line_3 = 'Anarchism'
#
#
# list_hype, txt, _, _, _, _ = extract_text_and_hyp(test_line_1, False)
# print(list_hype)
# print(txt)

# --------------------------------------------------------


def extract_page_entity_title(line, entity_name_id):
    start_off = line.find('<doc id="')
    end_off = start_off + len('<doc id="')
    assert start_off >= 0, line

    start_quotes = line.find('"', end_off)
    ent_wikiid = int(line[end_off: start_quotes])
    assert ent_wikiid, line + ' ==> ' + line[start_off: start_quotes]

    start_title_start_off = line.find(' title="')
    start_title_end_off = start_title_start_off + len(' title="')
    end_title_off = line.find('">')
    ent_name = line[start_title_end_off: end_title_off]
    if ent_wikiid != entity_name_id.get_ent_wikiid_from_name(ent_name, True):
        # Most probably this is a disambiguation or list page
        new_ent_wikiid = entity_name_id.get_ent_wikiid_from_name(ent_name, True)
        # print('Error in Wiki dump: {} {} {}'
        #       .format(line, ent_wikiid, new_ent_wikiid))
        return new_ent_wikiid, 1
    return ent_wikiid, 0
#
#
# test_line_4 = '<doc id="12" url="http://en.wikipedia.org/wiki?curid=12" title="Anarchism">'
#
# print(extract_page_entity_title(test_line_4))
