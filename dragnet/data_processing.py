from __future__ import division, print_function

import io
import itertools
import multiprocessing
import os
import re

import ftfy
from lxml import etree
import numpy as np

from dragnet.blocks import Blockifier, simple_tokenizer, text_from_subtree
from dragnet.compat import unicode_
from dragnet.lcs import check_inclusion


RAW_HTML_DIRNAME = 'HTML'
GOLD_STANDARD_DIRNAME = 'Corrected'
GOLD_STANDARD_BLOCKS_DIRNAME = 'block_corrected'

RAW_HTML_EXT = '.html'
GOLD_STANDARD_EXT = '.html.corrected.txt'
GOLD_STANDARD_BLOCKS_EXT = '.block_corrected.txt'

RE_COMMENTS_DELIM = re.compile(r'\n*!@#\$%\^&\*\(\)\s+COMMENTS\n*')


def extract_all_gold_standard_data(data_dir, nprocesses=1,
                                   overwrite=False, **kwargs):
    """
    Extract the gold standard block-level content and comment percentages from a
    directory of labeled data (only those for which the gold standard blocks are
    not found), and save results to corresponding files in a block-level
    gold standard directory under ``data_dir``.

    Args:
        data_dir (str): Directory on disk containing subdirectories for all
            training data, including raw html files and gold standard content +
            comments text files
        nprocesses (int): If > 1, use a :class:`multiprocessing.Pool` to
            parallelize the extractions
        overwrite (bool): If True, overwrite existing gold-standard blocks files.
        **kwargs: passed into :func:`extract_gold_standard_blocks`

    See Also:
        :func:`extract_gold_standard_blocks`
    """
    use_pool = nprocesses > 1
    if use_pool:
        pool = multiprocessing.Pool(processes=nprocesses)

    # get the set of files that have already been block corrected
    # so that we don't block correct them again
    if overwrite is False:
        gs_blocks_dir = os.path.join(data_dir, GOLD_STANDARD_BLOCKS_DIRNAME)
        if not os.path.isdir(gs_blocks_dir):
            os.mkdir(gs_blocks_dir)
        gs_blocks_filenames = get_filenames(
            gs_blocks_dir, full_path=False, match_regex=re.escape(GOLD_STANDARD_BLOCKS_EXT))
        gs_blocks_fileroots = {
            re.search(r'(.+)' + re.escape(GOLD_STANDARD_BLOCKS_EXT), gs_blocks_filename).group(1)
            for gs_blocks_filename in gs_blocks_filenames}
    else:
        gs_blocks_fileroots = set()

    # extract the block-level gold parse from
    # the set of files to be block corrected
    gs_dir = os.path.join(data_dir, GOLD_STANDARD_DIRNAME)
    gs_filenames = get_filenames(
        gs_dir, full_path=False, match_regex=re.escape(GOLD_STANDARD_EXT))
    for i, gs_filename in enumerate(gs_filenames):
        gs_fileroot = re.search(r'(.+)' + re.escape(GOLD_STANDARD_EXT), gs_filename).group(1)
        if gs_fileroot in gs_blocks_fileroots:
            continue
        if i % 100 == 0:
            print('Extracting gold standard blocks for file "{}"'.format(gs_filename))
        if use_pool:
            pool.apply_async(extract_gold_standard_blocks, (data_dir, gs_fileroot), kwargs)
        else:
            extract_gold_standard_blocks(data_dir, gs_fileroot, **kwargs)

    # close out our pool
    if use_pool:
        pool.close()
        pool.join()


def extract_gold_standard_blocks(data_dir, fileroot, encoding=None,
                                 tokenizer=simple_tokenizer, cetr=False):
    """
    Extract the gold standard block-level content and comments for a single
    observation identified by ``fileroot``, and write the results to file.

    Args:
        data_dir (str): The root directory containing sub-directories for
            raw HTML, gold standard extracted content, and gold standard blocks.
        fileroot (str): Unique identifier for a single observation of training
            data, corresponding to the start of its raw html and gold standard
            filenames under ``data_dir``.
        encoding (str)
        tokenizer (Callable): Object that takes a string and returns the tokens
            as a list of strings.
        cetr (bool): If True, parse the gold standard in clean eval format.

    Notes:
        Results are written to a text file in the block-level gold standard dir
        :obj:`GOLD_STANDARD_BLOCKS_DIRNAME` below ``data_dir``. Each line
        corresponds to a single block in its order of appearance, and has the
        following format::

            content_frac comments_frac all_tokens content_tokens comments_tokens

        where each item is separated by a tab. ``content_frac`` is equal to the
        fraction of ``all_tokens`` found in the corresponding gold parse content
        text; ``comments_frac`` is the same but for comments text.
    """
    # read the raw html, split it into blocks, and tokenize each block
    raw_html = read_html_file(data_dir, fileroot, encoding=encoding)  # text is unicode
    from dragnet.blocks import BlockifyError
    try:
        blocks = [b.text for b in Blockifier.blockify(raw_html)]  # text is bytes
    except BlockifyError as e:
        print('BlockifyError for file "{}"'.format(fileroot))
        return
    blocks_tokens = [tokenizer(block) for block in blocks]
    num_blocks_tokens = [len(block_tokens) for block_tokens in blocks_tokens]

    # solve the longest common subsequence problem to determine which blocks were kept
    # need a list of all the tokens in the blocks, plus a correspondence of which
    # block they belong to.
    # we will determine which of the tokens is in the extracted content,
    # then use the correspondence to block id to determine which blocks were kept

    # get a flattened sequence of all tokens in all blocks
    # and their corresponding block ids
    all_blocks_tokens = []
    all_blocks_tokens_block_id = []
    for i, block_tokens in enumerate(blocks_tokens):
        all_blocks_tokens.extend(block_tokens)
        all_blocks_tokens_block_id.extend([i] * len(block_tokens))
    # TODO: do we really need `num_all_blocks_tokens`?
    # it was used to determine if there were more gold standard tokens than *all*
    # tokens, and if so, some info was written to disk
    # but it seems like an odd check, and it's probably better to take the
    # gold standard data at face value -- presumably, somebody checked it!
    # num_all_blocks_tokens = len(all_blocks_tokens)

    def get_frac_and_str_tokens_in_gs(gs_txt):
        """
        For each block, determine which and what fraction of tokens are
        also in the gold standard text ``gs_txt`` for either content
        or comments.

        Returns:
            List[float]
            List[str]
        """
        gs_tokens = tokenizer(gs_txt)

        tokens_in_gs = check_inclusion(all_blocks_tokens, gs_tokens)
        num_blocks_tokens_in_gs = [0 for _ in range(len(blocks))]
        blocks_tokens_in_gs_tokens = [[] for _ in range(len(blocks))]
        for token, token_in_gs, block_id in zip(all_blocks_tokens, tokens_in_gs, all_blocks_tokens_block_id):
            if token_in_gs is True:
                num_blocks_tokens_in_gs[block_id] += 1
                blocks_tokens_in_gs_tokens[block_id].append(token)

        blocks_tokens_strs_in_gs = [
            ' '.join(block_tokens_in_gs_tokens)
            for block_tokens_in_gs_tokens in blocks_tokens_in_gs_tokens]
        frac_blocks_tokens_in_gs = [
            num_block_tokens_in_gs / num_block_tokens
            for num_block_tokens_in_gs, num_block_tokens
            in zip(num_blocks_tokens_in_gs, num_blocks_tokens)]

        return (frac_blocks_tokens_in_gs, blocks_tokens_strs_in_gs)

    gs_content, gs_comments = read_gold_standard_file(data_dir, fileroot, cetr)
    frac_blocks_tokens_in_gs_content, blocks_tokens_strs_in_gs_content = \
        get_frac_and_str_tokens_in_gs(gs_content)
    frac_blocks_tokens_in_gs_comments, blocks_tokens_strs_in_gs_comments = \
        get_frac_and_str_tokens_in_gs(gs_comments)

    output_fname = os.path.join(
        data_dir, GOLD_STANDARD_BLOCKS_DIRNAME, fileroot + GOLD_STANDARD_BLOCKS_EXT)
    line_fmt = u'{frac_content}\t{frac_comments}\t{block_tokens}\t{content_tokens}\t{comment_tokens}\n'
    with io.open(output_fname, mode='w') as f:
        for block_id, block_tokens in enumerate(blocks_tokens):
            line = line_fmt.format(
                frac_content=frac_blocks_tokens_in_gs_content[block_id],
                frac_comments=frac_blocks_tokens_in_gs_comments[block_id],
                block_tokens=' '.join(block_tokens),
                content_tokens=blocks_tokens_strs_in_gs_content[block_id],
                comment_tokens=blocks_tokens_strs_in_gs_comments[block_id])
            f.write(line)


def get_filenames(dirname, full_path=False, match_regex=None, extension=None):
    """
    Get all filenames under ``dirname`` that match ``match_regex`` or have file
    extension equal to ``extension``, optionally prepending the full path.

    Args:
        dirname (str): /path/to/dir on disk where files to read are saved
        full_path (bool): if False, return filenames without path; if True,
            return filenames with path, as ``os.path.join(dirname, fname)``
        match_regex (str): include files whose names match this regex pattern
        extension (str): if files only of a certain type are wanted,
            specify the file extension (e.g. ".txt")

    Yields:
        str: next matching filename
    """
    if not os.path.exists(dirname):
        raise OSError('directory "{}" does not exist'.format(dirname))
    match_regex = re.compile(match_regex) if match_regex else None
    for filename in sorted(os.listdir(dirname)):
        if extension and not os.path.splitext(filename)[-1] == extension:
            continue
        if match_regex and not match_regex.search(filename):
            continue
        if full_path is True:
            yield os.path.join(dirname, filename)
        else:
            yield filename


def read_html_file(data_dir, fileroot, encoding=None):
    """
    Read the HTML file corresponding to identifier ``fileroot``
    in the raw HTML directory below the root ``data_dir``.

    Args:
        data_dir (str)
        fileroot (str)
        encoding (str)

    Returns:
        str
    """
    fname = os.path.join(
        data_dir, RAW_HTML_DIRNAME, fileroot + RAW_HTML_EXT)
    encodings = (encoding,) if encoding else ('utf-8', 'iso-8859-1')  # 'utf-16'
    for encoding in encodings:
        try:
            with io.open(fname, mode='rt', encoding=encoding) as f:
                raw_html = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            raw_html = None

    return ftfy.fix_encoding(raw_html).strip()


def read_gold_standard_file(data_dir, fileroot, encoding=None, cetr=False):
    """
    Read the gold standard content file corresponding to identifier ``fileroot``
    in the gold standard directory below the root ``data_dir``.

    Args:
        data_dir (str)
        fileroot (str)
        encoding (str)
        cetr (bool): if True, assume no comments and parse the gold standard
            to remove tags

    Returns:
        List[str, str]: contents string and comments string, respectively
    """
    fname = os.path.join(
        data_dir, GOLD_STANDARD_DIRNAME, fileroot + GOLD_STANDARD_EXT)
    encodings = (encoding,) if encoding else ('utf-8', 'utf-16', 'iso-8859-1')
    for encoding in encodings:
        try:
            with io.open(fname, mode='rt', encoding=encoding) as f:
                gold_standard = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            gold_standard = None

    if not gold_standard:
        return [u'', u'']

    if not cetr:
        content_comments = RE_COMMENTS_DELIM.split(gold_standard, maxsplit=1)
        # if no comments delimiter found, append empty comments string
        if len(content_comments) == 1:
            content_comments = [content_comments[0], u'']
    else:
        tree = etree.fromstring(gold_standard, parser=etree.HTMLParser())
        content_comments = [u' '.join(text_from_subtree(tree)), u'']

    # fix text in case of mangled encodings
    content_comments = [ftfy.fix_encoding(content_comments[0]).strip(),
                        ftfy.fix_encoding(content_comments[1]).strip()]

    return content_comments


def read_gold_standard_blocks_file(data_dir, fileroot, split_blocks=True):
    """
    Read the gold standard blocks file corresponding to identifier ``fileroot``
    in the gold standard blocks directory below the root ``data_dir``.

    Args:
        data_dir (str)
        fileroot (str)
        split_blocks (bool): If True, split the file's content into blocks.

    Returns:
        str or List[str]
    """
    fname = os.path.join(
        data_dir, GOLD_STANDARD_BLOCKS_DIRNAME, fileroot + GOLD_STANDARD_BLOCKS_EXT)
    with io.open(fname, mode='r') as f:
        data = f.read()
    if split_blocks:
        return filter(None, data[:-1].split('\n'))
    return filter(None, data)


def _parse_content_or_comments_blocks(blocks, block_pct_tokens_thresh):
    is_above_thresh = (np.array([ele[0] for ele in blocks]) > block_pct_tokens_thresh).astype(np.int)
    token_counts = np.array([ele[1] for ele in blocks])
    all_tokens = list(itertools.chain.from_iterable(
        ele[2] for ele in blocks if ele[1] > 0))
    return (is_above_thresh, token_counts, all_tokens)


def prepare_data(data_dir, fileroot, block_pct_tokens_thresh=0.1):
    """
    Prepare data for a single HTML + gold standard blocks example, uniquely
    identified by ``fileroot``.

    Args:
        data_dir (str)
        fileroot (str)
        block_pct_tokens_thresh (float): must be in [0.0, 1.0]

    Returns:
        Tuple[str, Tuple[np.array[int], np.array[int], List[str]], Tuple[np.array[int], np.array[int], List[str]]]:
            The first element is simply the raw html as a string. The second and
            third elements are 3-tuples for content and comments, respectively,
            where the first element is a numpy array of 1s and 0s whose values
            correspond to whether or not a given block is considered non-content
            or not; the second element is a numpy integer array whose values are
            the total number of tokens in each block; and the third element is
            a flat list of content or comment tokens as strings, concatenated
            from all blocks.

    See Also:
        :func:`prepare_all_data`
    """
    if not 0.0 <= block_pct_tokens_thresh <= 1.0:
        raise ValueError('block_pct_tokens_thresh must be in the range [0.0, 1.0]')

    html = read_html_file(data_dir, fileroot)
    blocks = read_gold_standard_blocks_file(data_dir, fileroot, split_blocks=True)

    content_blocks = []
    comments_blocks = []
    for block in blocks:
        block_split = block.split('\t')
        num_block_tokens = len(block_split[2].split())
        # total number of tokens in block is used as weights
        content_blocks.append(
            (float(block_split[0]), num_block_tokens, block_split[3].split()))
        comments_blocks.append(
            (float(block_split[1]), num_block_tokens, block_split[4].split()))

    parsed_content_blocks = _parse_content_or_comments_blocks(
        content_blocks, block_pct_tokens_thresh)
    parsed_comments_blocks = _parse_content_or_comments_blocks(
        comments_blocks, block_pct_tokens_thresh)

    return (html, parsed_content_blocks, parsed_comments_blocks)


def prepare_all_data(data_dir, block_pct_tokens_thresh=0.1):
    """
    Prepare data for all HTML + gold standard blocks examples in ``data_dir``.

    Args:
        data_dir (str)
        block_pct_tokens_thresh (float): must be in [0.0, 1.0]

    Returns:
        List[Tuple[str, List[float, int, List[str]], List[float, int, List[str]]]]

    See Also:
        :func:`prepare_data`
    """
    gs_blocks_dir = os.path.join(data_dir, GOLD_STANDARD_BLOCKS_DIRNAME)
    gs_blocks_filenames = get_filenames(
        gs_blocks_dir, full_path=False, match_regex=re.escape(GOLD_STANDARD_BLOCKS_EXT))
    gs_blocks_fileroots = (
        re.search(r'(.+)' + re.escape(GOLD_STANDARD_BLOCKS_EXT), gs_blocks_filename).group(1)
        for gs_blocks_filename in gs_blocks_filenames)

    return [prepare_data(data_dir, fileroot, block_pct_tokens_thresh)
            for fileroot in gs_blocks_fileroots]

