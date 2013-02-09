
import re
import json
import numpy as np
import pylab as plt
import glob
import codecs

from .blocks import Blockifier, simple_tokenizer

def read_gold_standard(datadir, fileroot):
    """Reads the gold standard from the disk.
    Returns [content, comments] where content/comments are strings"""
    from chardet.universaldetector import UniversalDetector

    corrected_file = datadir + '/Corrected/%s.html.corrected.txt' % fileroot
    def read_file(encoding):
        return codecs.open(corrected_file, 'r', encoding=encoding).read()

    try:
        gold_standard = read_file('utf-8')
    except UnicodeDecodeError:
        try:
            gold_standard = read_file('utf-16')
        except UnicodeError:
            gold_standard = read_file('iso-8859-1')

    # split gold_standard into content and comments
    # use an array so we can iterate over it
    content_comments = re.split(r'!@#\$%\^&\*\(\)\s+COMMENTS', gold_standard)
    if len(content_comments) == 1:
        # no comments
        return [content_comments[0], '']
    else:
        return content_comments


def get_list_all_corrected_files(datadir):
    """
        Given datadir, return a list of tuples
            (file, fileroot)
        for all the corrected files
    """
    ret = []
    files = glob.glob(datadir + "/Corrected/*")
    for file in files:
        mo = re.search('Corrected\/(.+)\.html\.corrected\.txt$', file)
        fileroot = mo.group(1)
        ret.append((file, fileroot))
    return ret


def extract_gold_standard(datadir, fileroot,
                          tokenizer=simple_tokenizer, cleaneval=False):
    """datadir = the root datadir.
            Contains sub-directories: HTML, Corrected, block_corrected
       tokenizer = callable object that takes a string and returns the tokens
            as a list of strings
       cleaneval = if true, then parse the gold standard in clean eval format

       Input data files:
            HTML/fileroot.html
            Corrected/fileroot.html.corrected.txt
        Output data file:
            block_corrected/fileroot.block_corrected.txt
            Contains:
                content_percent comments_percent all_tokens_in_block content_tokens comment_tokens
            each separated by a \t
            each line contains info for a single block

            for the percent:
                -1 if block has no content (so no prediction = not content)
                0-1 float that is the percent of block tokens that are content

       Uses blockify, the tokenizer, lcs
       """
    # NOTE: this code was written when the definition return from
    # Blockifier.blockify returned blocks without any text
    # content.  As such, it must handle blocks without text as a special
    # case in the code below.
    # Additionally, the resulting output file uses the flag -1 to represent
    # empty blocks.
    #
    # blockify has since been updated, so this logic isn't needed below.
    # however, it doesn't change the output so it has been left
    # alone to avoid unnecessary code modifications

    from .lcs import check_inclusion
    from .blocks import PartialBlock
    from lxml import etree

    # get the raw content, split it into blocks, tokenize
    raw_content = open(datadir + '/HTML/%s.html' % fileroot, 'r').read()
    blocks = [b.text for b in Blockifier.blockify(raw_content)]

    # blocks_tokens = a list of tokens in each block
    # contains '' if the block contains no tokens
    blocks_tokens = []
    i = 0
    for block in blocks:
        i += 1
        if len(block.strip()) == 0:
            blocks_tokens.append('')
        else:
            blocks_tokens.append(tokenizer(block))

    # solve the longest common subsequence problem to determine which blocks were kept
    # need a list of all the tokens in the blocks, plus a correspondence of which
    # block they belong to.
    # we will determine which of the tokens is in the extracted content,
    # then use the correspondence to block id to determine which blocks were kept
    all_blocks_tokens = []           # a list of tokens in the document
    all_blocks_tokens_block_id = []  # the corresponding block id
    i = 0
    for block in blocks_tokens:
        if block != '':
            all_blocks_tokens.extend(block)
            all_blocks_tokens_block_id.extend([i] * len(block))
        i += 1


    gold_standard_content_comments = read_gold_standard(datadir, fileroot)
    content_comments = ['content', 'comments']

    content_comments_tokens = []
    content_comments_percent = []
    for k in xrange(len(gold_standard_content_comments)):

        # if it is a cleaneval dataset, need to parse and extact text
        if cleaneval:
            if k == 0:
                if len(gold_standard_content_comments[0].strip()) > 0:
                    tree = etree.fromstring(gold_standard_content_comments[0], parser=etree.HTMLParser())
                    txt = ' '.join(PartialBlock._text_from_subtree(tree))
                else:
                    txt = ''
            else:
                # no comments for cleaneval
                txt = ''
        else:
            txt = gold_standard_content_comments[k]

        gold_standard_tokens = tokenizer(txt)
    
        print "Got all tokens for %s.  %s in all blocks, %s in gold standard %s" % (fileroot, len(all_blocks_tokens), len(gold_standard_tokens), content_comments[k])
        #tokens_in_gold_standard = check_inclusion(all_blocks_tokens, gold_standard_tokens)
        tokens_in_gold_standard = check_inclusion(
                [t.encode('utf-8') for t in all_blocks_tokens], 
                [t.encode('utf-8') for t in gold_standard_tokens])
    
        # now make a percentage of tokens in the gold standard for each block
        blocks_token_count = [0] * len(blocks)
        blocks_tokens_in_gold_standard = [0] * len(blocks)
        blocks_tokens_in_gold_standard_tokens = [' '] * len(blocks)
        for block in zip(tokens_in_gold_standard, all_blocks_tokens_block_id, all_blocks_tokens):
            blocks_token_count[block[1]] += 1
            if block[0]:
                blocks_tokens_in_gold_standard[block[1]] += 1
                blocks_tokens_in_gold_standard_tokens[block[1]] += block[2] + ' '
    
        # the array of block level token percent
        token_percent = [-1 if ele[0] == 0 else float(ele[1]) / ele[0] for ele in zip(blocks_token_count, blocks_tokens_in_gold_standard)]

        content_comments_tokens.append(blocks_tokens_in_gold_standard_tokens)
        content_comments_percent.append(token_percent)


    # write the final output file
    with codecs.open(datadir + '/block_corrected/%s.block_corrected.txt' % fileroot, 'w', encoding='utf-8') as f:
        for block in zip(content_comments_percent[0], content_comments_percent[1], blocks_tokens, content_comments_tokens[0], content_comments_tokens[1]):
            f.write(str(block[0]) + '\t' + str(block[1]) + '\t')
            if block[2] == '':
                f.write(' \t')
            else:
                f.write(' '.join(block[2]) + '\t')
            f.write(block[3] + '\t' + block[4] + '\n')
            k += 1


def extract_gold_standard_all_training_data(datadir, nprocesses=40, **kwargs):
    """
        Extract the gold standard block level content and comment
        percentages from a directory of labeled data
        if nprocesses > 1, then use a process pool
        **kwargs passed into extract_gold_standard
    """
    use_pool = nprocesses > 1
    if use_pool:
        from multiprocessing import Pool
        p = Pool(processes=nprocesses)

    # get a list of files that have already been block corrected
    # don't block correct them again
    files_already_block_corrected = glob.glob(datadir + "/block_corrected/*")
    fileroot_already_corrected = set([re.search("block_corrected\/(.+)\.block_corrected", ele).group(1) for ele in files_already_block_corrected])

    # all the corrected files
    for file, fileroot in get_list_all_corrected_files(datadir):
        if fileroot not in fileroot_already_corrected:
            print "Extracting gold standard for file %s" % fileroot
            if use_pool:
                p.apply_async(extract_gold_standard, (datadir, fileroot), kwargs)
            else:
                extract_gold_standard(datadir, fileroot, **kwargs)

    if use_pool:
        p.close()
        p.join()



class DragnetModelData(object):
    """
    the data needed to train a model
    includes the html, the gold standard tokens

    a datadir with the training data directory structure
    each training data document has a number of files with a common "fileroot" and
    a set of additional files in subdirectories
        HTML / fileroot.html
        Corrected / fileroot.html.corrected.txt = cut and paste content from the HTML
        block_corrected / fileroot.block_corrected.txt
    source = one of 'all', 'domain_list', 'technoratti', 'reader'
    """
    def __init__(self, datadir, block_percent_threshold=0.1, source='all'):
        # set the re_source = a regex that can be used on fileroot
        # to eliminate files based on source
        if source == 'technoratti':
            re_keep = '^T[0-9]+'
        elif source == 'domain_list':
            re_keep = '(^[0-9])|(^[a-zA-Z]{2})'
        elif source == 'reader':
            re_keep = '^R[0-9]+'
        elif source == 'all':
            re_keep = ''  # match anything
        else:
            raise InputError, "Invalid source"
        self._re_source = re.compile(re_keep)
        self._source = source

        # now read in all the data
        self._read_all_data(datadir, block_percent_threshold, source)


    def _read_all_data(self, datadir, block_percent_threshold, source):
        """
        block_percent_threshold = the cut-off percent of all tokens in a block
            that are in the gold standard, above which the block is classified as content
        stores attributes .training_data, .test_data where each is a list of tuples:
            (raw_html_string, content_gold_standard, comments_gold_standard)
            where content/comments gold_standard = (list of block 0/1 flag, list of # tokens, all tokens as a list)
        stores attributes .training_files, .test_files where each is a list
            of the file names
        """
        self.training_data = []
        self.test_data = []
        self.training_files = []
        self.test_files = []

        training_fileroot = set(open(datadir + '/training.txt', 'r').read().strip().split())
        print("Reading the training and test data...")
        for file, fileroot in get_list_all_corrected_files(datadir):
            if self._re_source.match(fileroot):
                html = open(datadir + '/HTML/%s.html' % fileroot, 'r').read()
                block_corrected_file = codecs.open(datadir + '/block_corrected/%s.block_corrected.txt' % fileroot, 'r', encoding='utf-8')
                blocks = block_corrected_file.read()[:-1].split('\n')
    
                content = []
                comments = []
                for block in blocks:
                    block_split = block.split('\t')
                    # will store the weights as the total number of tokens in the document
                    content.append((float(block_split[0]), len(block_split[2].strip().split()), block_split[3].strip().split()))
                    comments.append((float(block_split[1]), len(block_split[2].strip().split()), block_split[4].strip().split()))
    
                ret = []
                for content_comments in [content, comments]:
                    extracted_flag = (np.array([ele[0] for ele in content_comments]) > block_percent_threshold).astype(np.int)
                    extracted_flag[np.array([ele[0] for ele in content_comments]) == -1] = -1
                    counts = np.array([ele[1] for ele in content_comments])
                    tokens = []
                    for this_block_tokens in [ele[2] for ele in content_comments if ele[1] > 0]:
                        tokens.extend(this_block_tokens)
                    ret.append((extracted_flag, counts, tokens))
    
                if fileroot in training_fileroot:
                    self.training_data.append((html, ret[0], ret[1]))
                    self.training_files.append(fileroot)
                else:
                    self.test_data.append((html, ret[0], ret[1]))
                    self.test_files.append(fileroot)

        print("..done!")
        print("Got %s training, %s test documents" % (len(self.training_data), len(self.test_data)))


    @staticmethod
    def diagnose_css(datadir, plotdir):
        data = DragnetModelData(datadir, source='all')

        # get a list of all the css tokens extracted as content and not content
        # ONLY USE TRAINING DATA
        content_css = []
        no_content_css = []
        for datum in data.training_data:
            blocks = Blockifier.blockify(datum[0])
            extracted = np.logical_or(datum[1][0], datum[2][0])
            assert len(blocks) == len(extracted)
            content_css.extend([blocks[k].css for k in xrange(len(blocks)) if extracted[k]])
            no_content_css.extend([blocks[k].css for k in xrange(len(blocks)) if not extracted[k]])

        # make a list of the most popular tokens
        from collections import defaultdict
        popular_tokens = {}
        for c, d in [('content', content_css), ('no_content', no_content_css)]:
            popular_tokens[c] = {}
            for tag in ['id', 'class']:
                popular_tokens[c][tag] = defaultdict(lambda: 0)
            for block in d:
                for tag in ['id', 'class']:
                    for token in re.split('\W+|_', block[tag]):
                        popular_tokens[c][tag][token] += 1

        # sort tokens by most popular
        popular_tokens_sorted = {}
        for c in ['content', 'no_content']:
            popular_tokens_sorted[c] = {}
            for tag in ['id', 'class']:
                popular_tokens_sorted[c][tag] = [(v, k) for k, v in popular_tokens[c][tag].iteritems()]
                popular_tokens_sorted[c][tag].sort(reverse=True)


        # write to a file with percent of total
        for c in ['content', 'no_content']:
            for tag in ['id', 'class']:
                total_tokens = np.sum([ele[0] for ele in popular_tokens_sorted[c][tag]])
                with open(plotdir + '/css_token_count_%s_%s.tsv' % (c, tag), 'w') as f:
                    f.write("Token\tCount\tPercent Total\tCum Total\n")
                    cumcount = 0
                    for count, token in popular_tokens_sorted[c][tag]:
                        cumcount += count
                        f.write("%s\t%s\t%s\t%s\n" %
                                    (count,
                                     token,
                                     float(count)/total_tokens,
                                     float(cumcount) / total_tokens))

        # take the ratio of token count in content vs no content
        # for the tokens in the specified list
        css_tokens = open("dragnet_css_tokens.txt", 'r').read().strip().split('\n')
        content_no_content_ratio = {}
        no_content_block_count = len(no_content_css)
        content_block_count = len(content_css)
        for tag in ['id', 'class']:
            content_no_content_ratio[tag] = []
            for token in css_tokens:
                content_count_percent = np.sum([re.search(token, block[tag].lower()) is not None for block in content_css]) / float(content_block_count)
                no_content_count_percent = np.sum([re.search(token, block[tag].lower()) is not None for block in no_content_css]) / float(no_content_block_count)

                if no_content_count_percent > 0:
                    ratio = content_count_percent / no_content_count_percent
                else:
                    ratio = np.inf

                content_no_content_ratio[tag].append((ratio, token, content_count_percent, no_content_count_percent))

            content_no_content_ratio[tag].sort()

        # dump ratios to a file
        with open(plotdir + '/css_popular_token_ratio.txt', 'w') as f:
            f.write("Ratio of appearence frequency in content vs non-content blocks\n")
            f.write("Ratio------token-----percent of content blocks present-----percent of non-content blocks present\n")
            for tag in ['id', 'class']:
                f.write("\n%s\n" % tag)
                for t in content_no_content_ratio[tag]:
                    f.write("%s\t%s\t%s\t%s\n" % t)



    @staticmethod
    def diagnose_data(datadir, plotdir, training_or_test='both'):
        """Do some diagnosis if the data set

        Plotdir = output plots to this directory"""

        # we will accumulate the percent extracted for some histograms
        percent_extracted = []
        for s, t in [('all', 'All data'),
                    ('technoratti', 'Technoratti'),
                    ('domain_list', "Domain list"),
                    ('reader', "Popular RSS on Google Reader")]:

            data = DragnetModelData(datadir, source=s)
            data._diagnose_data_one_source(plotdir, t, training_or_test='both')

            percent_extracted.append((t, data._get_percent_tokens_extracted_in_block(datadir)))

        # plot percent extracted
        fig = plt.figure(3)
        fig.clf()
        k = 0
        for ti, d in percent_extracted:
            plt.subplot(221 + k)
            plt.hist(d, 30)
            plt.title(ti)
            k += 1
        fig.show()
        fig.savefig(plotdir + '/percent_tokens_extracted.png')


    def _get_percent_tokens_extracted_in_block(self, datadir):
        ret = []
        for file, fileroot in get_list_all_corrected_files(datadir):
            if self._re_source.match(fileroot):
                # a histogram of block frequency
                block_corrected_file = codecs.open(datadir + '/block_corrected/%s.block_corrected.txt' % fileroot, 'r', encoding='utf-8')
                blocks = block_corrected_file.read()[:-1].split('\n')

                for block in blocks:
                    block_split = block.split('\t')
                    ret.append(float(block_split[0]))

        return np.asarray(ret)


    def _diagnose_data_one_source(self, plotdir, ti, training_or_test='both'):
        """Make some plots and do some exploratory analyis on training data
        training_or_test is one of "training", "test", "both"
        """
        from mozsci.histogram import Histogram1DFast
        if training_or_test == 'training':
            plot_data = self.training_data
            files = self.training_files
        elif training_or_test == 'test':
            plot_data = self.test_data
            files = self.test_files
        elif training_or_test == 'both':
            plot_data = self.training_data + self.test_data
            files = self.training_files + self.test_files
        else:
            raise InputError, "Invalid training_or_test"

        # block_level_aggreate = holds block count of # extracted as
        #                        content, comments and total
        block_level_aggregate = {'content':[], 'comments':[], 'total':[]}
        for datum in plot_data:
            k = 1
            block_level_aggregate['total'].append(len(datum[1][1]))
            for c in ['content', 'comments']:
                extracted_flag, overall_token_count, tokens = datum[k]
                block_level_aggregate[c].append(np.sum(extracted_flag))
                k += 1

        # plot
        block_level_aggregate['total'] = np.array(block_level_aggregate['total']).astype(np.float)
        fig = plt.figure(1)
        fig.clf()

        plt.subplot(221)
        plt.hist(block_level_aggregate['total'], 30)
        plt.title("Block count across files")

        plt.subplot(222)
        plt.hist(block_level_aggregate['content'] / block_level_aggregate['total'], 30)
        plt.title("Percent of blocks that are content across files")

        plt.subplot(223)
        plt.hist(block_level_aggregate['comments'] / block_level_aggregate['total'], 30)
        plt.title("Percent of blocks that are comments across files")

        txt = "Total blocks: %s " % int(np.sum(block_level_aggregate['total']))
        for s in ['content', 'comments']:
            txt += "\nTotal %s %s (%s %%)" % (s, int(np.sum(block_level_aggregate[s])), np.sum(block_level_aggregate[s]) / np.sum(block_level_aggregate['total']) * 100)
        plt.figtext(0.6, 0.4, txt)
        
        add_plot_title(ti + '\nBlock level, training + test')

        fig.show()
        fig.savefig(plotdir + '/' + self._source + '_block_level.png')

        # percent extracted as content vs block number
        bins = 20
        content_percent_vs_block_percent = {'content':np.zeros((len(plot_data), bins)),
                                            'comments':np.zeros((len(plot_data), bins))}

        # number of tokens in block vs block number
        block_length_vs_block_percent = np.zeros((len(plot_data), bins))

        for datum_number in xrange(len(plot_data)):
            datum = plot_data[datum_number]
            k = 1
            for c in ['content', 'comments']:
                extracted_flag, overall_token_count, tokens = datum[k]
                block_percent = np.arange(len(extracted_flag)) / float(len(extracted_flag))

                # count of extracted blocks in each bin
                h = Histogram1DFast(bins, 0, 1)
                h.update_counts(block_percent, extracted_flag)
                extracted_counts = h.bin_count

                # overall count
                h = Histogram1DFast(bins, 0, 1)
                h.update(block_percent)
                total_counts = h.bin_count

                # number of tokens in block
                if c == 'content':  # token count same for content, comments
                    h = Histogram1DFast(bins, 0, 1)
                    h.update_counts(block_percent, overall_token_count)
                    token_count = h.bin_count
                    block_length_vs_block_percent[datum_number, :] = token_count.astype(np.float) / total_counts

                content_percent_vs_block_percent[c][datum_number, :] = extracted_counts.astype(np.float) / total_counts
                k += 1

        # plot
        fig = plt.figure(2)
        fig.clf()

        plt.subplot(311)
        c = 'content'
        masked_data = np.ma.masked_array(content_percent_vs_block_percent[c], np.isnan(content_percent_vs_block_percent[c]))
        np.mean(masked_data, axis=0)
        plt.plot(np.linspace(0, 1, bins), np.mean(masked_data, axis=0))
        plt.title("Content")
        plt.ylabel("Percent extracted")

        plt.subplot(312)
        c = 'comments'
        masked_data = np.ma.masked_array(content_percent_vs_block_percent[c], np.isnan(content_percent_vs_block_percent[c]))
        np.mean(masked_data, axis=0)
        plt.plot(np.linspace(0, 1, bins), np.mean(masked_data, axis=0))
        plt.title("Comments")
        plt.ylabel("Percent extracted")

        plt.subplot(313)
        masked_data = np.ma.masked_array(block_length_vs_block_percent, np.isnan(block_length_vs_block_percent))
        np.mean(masked_data, axis=0)
        plt.plot(np.linspace(0, 1, bins), np.mean(masked_data, axis=0))
        plt.title("All tokens")
        plt.xlabel("Block position in document")
        plt.ylabel("# tokens in block")

        add_plot_title(ti + '\nPercent of blocks extracted, # tokens in doc, training + test')
        fig.show()
        fig.savefig(plotdir + '/' + self._source + '_block_level_block_position.png')


def split_data(datadir):
    """Split the data into training/test sets.
    write files containing test and training data file roots
    test = 30% of data
    """
    from random import shuffle
    all_files = get_list_all_corrected_files(datadir)
    shuffle(all_files)
    nfiles = len(all_files)
    ntrain = int(nfiles * 0.7)

    # write training/test lists
    open(datadir + '/training.txt', 'w').write('\n'.join([ele[1] for ele in all_files[:ntrain]]))
    open(datadir + '/test.txt', 'w').write('\n'.join([ele[1] for ele in all_files[ntrain:]]))

