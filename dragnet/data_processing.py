from __future__ import division, print_function

import collections
import io
import itertools
import logging
import multiprocessing
import os
import re

import ftfy
import numpy as np
from lxml import etree
from tqdm import tqdm

from dragnet.blocks import Blockifier, simple_tokenizer, text_from_subtree
from dragnet.lcs import check_inclusion

COMMENTS_DELIM = "*!@#$%^&*() COMMENTS"
RE_COMMENTS_DELIM = re.compile(r"\n{}\n*".format(re.escape(COMMENTS_DELIM)))

HTML_DIRNAME = "html"
TEXT_DIRNAME = "text"
BLOCKS_DIRNAME = "blocks"

HTML_EXT = ".html"
TEXT_EXT = ".txt"
BLOCKS_EXT = ".blocks.txt"

RE_RECORD_FNAME = re.compile(
    r"^(?P<record_id>\w+)(?P<ext>{})$".format(
        "|".join(re.escape(ext) for ext in (HTML_EXT, TEXT_EXT, BLOCKS_EXT))),
    flags=re.UNICODE,
)


def set_dirname(html=None, text=None, blocks=None):
    if not (html or text or blocks):
        raise ValueError("at least one of html, text, or blocks must be specified")
    if html:
        global HTML_DIRNAME
        HTML_DIRNAME = html
    if text:
        global TEXT_DIRNAME
        TEXT_DIRNAME = text
    if blocks:
        global BLOCKS_DIRNAME
        BLOCKS_DIRNAME = blocks


def set_extension(html=None, text=None, blocks=None):
    if not (html or text or blocks):
        raise ValueError("at least one of html, text, or blocks must be specified")
    if html:
        global HTML_EXT
        HTML_EXT = html
    if text:
        global TEXT_EXT
        TEXT_EXT = text
    if blocks:
        global BLOCKS_EXT
        BLOCKS_EXT = blocks


class Record(object):
    """
    Simple class to consistently get record identifiers and filepaths for their
    html/text/blocks data, without having to pass around a bunch of constants.

    Args:
        record_id (str): Alphanumeric string used to uniquely identify a record.
        root_dir (str): Path to root directory directly under which sub-dirs
            for HTML, gold-standard text, and gold-standard blocks data is stored.

    Attributes:
        record_id (str)
        root_dir (str)
        html_fpath (str)
        text_fpath (str)
        blocks_fpath (str)
    """

    def __init__(self, record_id, root_dir=None):
        self.record_id = record_id
        self.root_dir = root_dir

    def __str__(self):
        return "Record({})".format(self.record_id)

    @classmethod
    def from_fname(cls, fname):
        match = RE_RECORD_FNAME.search(fname)
        if match:
            return cls(match.group("record_id"))
        else:
            raise ValueError("unable to get record id from file '{}'".format(fname))

    @classmethod
    def from_fpath(cls, fpath):
        fname = os.path.basename(fpath)
        dirname = os.path.dirname(fpath)
        root_dir = os.path.dirname(dirname)
        match = RE_RECORD_FNAME.search(fname)
        if match:
            return cls(match.group("record_id"), root_dir=root_dir)
        else:
            raise ValueError("unable to get record id from file '{}'".format(fpath))

    def _get_fpath(self, dirname, ext):
        if self.root_dir:
            return os.path.join(self.root_dir, dirname, self.record_id + ext)
        else:
            return None

    @property
    def html_fpath(self):
        return self._get_fpath(HTML_DIRNAME, HTML_EXT)

    @property
    def text_fpath(self):
        return self._get_fpath(TEXT_DIRNAME, TEXT_EXT)

    @property
    def blocks_fpath(self):
        return self._get_fpath(BLOCKS_DIRNAME, BLOCKS_EXT)


def generate_all_gold_standard_blocks(root_dir, overwrite=False, **kwargs):
    """
    Generate gold-standard block-level data from a directory of HTML files with
    gold-standard text extractions, and save results into like-named files
    in another directory under ``root_dir``.

    Args:
        root_dir (str): Directory on disk containing sub-directories for
            training data: raw html files, gold standard text files, and
            gold standard blocks files.
        overwrite (bool): If True, generate blocks data for all text files and
            overwrite any blocks files that may exist; otherwise, only generate
            blocks data for those text files without corresponding blocks files.
        **kwargs: passed as-is into :func:`generate_gold_standard_blocks()`

    See Also:
        :func:`generate_gold_standard_blocks()`
    """
    # get the set of records that have already been block corrected
    # so that we don't block correct them again
    if overwrite is False:
        blocks_dirpath = os.path.join(root_dir, BLOCKS_DIRNAME)
        if not os.path.isdir(blocks_dirpath):
            raise OSError("blocks directory '{}' does not exist".format(blocks_dirpath))
        blocks_fnames = get_filenames(
            blocks_dirpath, full_path=False, match_regex=RE_RECORD_FNAME)
        blocks_record_ids = {
            Record.from_fname(fname).record_id
            for fname in blocks_fnames}
    else:
        blocks_record_ids = set()
    # extract the block-level gold parse from
    # the set of files to be block corrected
    text_dirpath = os.path.join(root_dir, TEXT_DIRNAME)
    text_fnames = list(
        get_filenames(text_dirpath, full_path=False, match_regex=RE_RECORD_FNAME)
    )
    with tqdm(total=len(text_fnames)) as pbar:
        for fname in text_fnames:
            pbar.update(1)
            record = Record.from_fname(fname)
            record_id = record.record_id
            if overwrite is False and record_id in blocks_record_ids:
                continue
            try:
                generate_gold_standard_blocks(record_id, root_dir, **kwargs)
            except Exception:
                logging.exception(
                    "unable to extract gold-standard blocks for %s", record)


def generate_gold_standard_blocks(
    record_id,
    root_dir,
    tokenizer=simple_tokenizer,
    cleaneval=False,
):
    """
    Generate gold standard block-level data for a single record in the training
    dataset identified by ``record_id``, and save results to a like-named file
    in another directory under ``root_dir``.

    Args:
        record_id (str): Unique identifier for a single record in the training
            dataset, corresponding to the portion of its html / gold standard
            filename before the file extension.
        root_dir (str): Directory on disk containing sub-directories for
            training data: raw html files, gold standard text files, and
            gold standard blocks files.
        tokenizer (Callable): Object that takes a text (str) and splits it
            into tokens (List[str]).
        cleaneval (bool): If True, parse the gold standard text in CleanEval format.

    Notes:
        Results are written to a text file in the block-level gold standard directory
        :obj:`BLOCKS_DIRNAME` below ``root_dir``. Each line corresponds to
        a single block in order of appearance, and has the following format::

            frac_tokens_in_text frac_tokens_in_comments all_tokens tokens_in_text tokens_in_comments

        with each item separated by a tab. ``frac_tokens_in_text`` is equal to the
        fraction of ``all_tokens`` found in the gold-standard text.

    Todo:
        get rid of the comments stuff
    """
    record = Record(record_id, root_dir=root_dir)
    # load the html and split it into blocks
    # text is unicode
    html = load_html_data(record.html_fpath)
    from dragnet.blocks import BlockifyError
    try:
        # text is bytes
        blocks = [b.text for b in Blockifier.blockify(html)]
    except BlockifyError as e:
        logging.exception("unable to blockify html file for record %s", record_id)
        return
    # load the text, and parse as needed
    text = load_text_data(record.text_fpath, cleaneval=cleaneval)
    blocks_data = _compute_blocks_data(text, blocks, tokenizer)
    # TODO: totally rip the comments elements out of each line, as below
    # line_fmt = u"{frac_tokens_in_text}\t{tokens}\t{tokens_in_text}\n"
    line_fmt = u"{frac_tokens_in_text}\t0.0\t{tokens}\t{tokens_in_text}\t\n"
    with io.open(record.blocks_fpath, mode="wt", encoding="utf-8") as f:
        for block_data in blocks_data:
            f.write(line_fmt.format(**block_data))


def _compute_blocks_data(text, blocks, tokenizer):
    """
    For each block in ``blocks``, determine which and what fraction of tokens are
    also found in the gold standard ``text``.

    Args:
        text (str)
        blocks (List[str])
        tokenizer (Callable)

    Returns:
        List[dict]
    """
    blocks_tokens = [tokenizer(block) for block in blocks]
    all_tokens = [token for tokens in blocks_tokens for token in tokens]
    all_token_in_texts = check_inclusion(all_tokens, tokenizer(text))
    all_token_block_ids = [
        block_id
        for block_id, tokens in enumerate(blocks_tokens)
        for _ in range(len(tokens))
    ]
    tokens_in_text = collections.defaultdict(list)
    for token, in_text, block_id in zip(all_tokens, all_token_in_texts, all_token_block_ids):
        if in_text is True:
            tokens_in_text[block_id].append(token)
    frac_tokens_in_text = {
        block_id: len(tokens_in_text.get(block_id, [])) / len(blocks_tokens[block_id])
        for block_id in range(len(blocks))
    }
    return [
        {
            "frac_tokens_in_text": frac_tokens_in_text[block_id],
            "tokens": " ".join(blocks_tokens[block_id]),
            "tokens_in_text": " ".join(tokens_in_text[block_id]),
        }
        for block_id in range(len(blocks))
    ]


def prepare_all_training_data(root_dir, min_frac_tokens=0.2):
    """
    Prepare training data for all HTML + gold standard blocks records in ``root_dir``.

    Args:
        root_dir (str)
        min_frac_tokens (float): Minimum fraction of block tokens found
            in gold-standard text extraction for the block to be considered "content".
            Value must be in [0.0, 1.0].

    Returns:
        List[Tuple[str, List[float, int, List[str]], List[float, int, List[str]]]]

    See Also:
        :func:`prepare_training_data()`
    """
    if not 0.0 <= min_frac_tokens <= 1.0:
        raise ValueError("`min_frac_tokens` must be in [0.0, 1.0]")
    blocks_dirpath = os.path.join(root_dir, BLOCKS_DIRNAME)
    if not os.path.isdir(blocks_dirpath):
        raise OSError("blocks directory '{}' does not exist".format(blocks_dirpath))
    blocks_fnames = get_filenames(
        blocks_dirpath, full_path=False, match_regex=RE_RECORD_FNAME)
    blocks_record_ids = [
        Record.from_fname(fname).record_id
        for fname in blocks_fnames]
    return [
        prepare_training_data(record_id, root_dir, min_frac_tokens)
        for record_id in blocks_record_ids]


def prepare_training_data(record_id, root_dir, min_frac_tokens=0.2):
    """
    Prepare data for a single HTML + gold standard blocks example, uniquely
    identified by ``record_id``.

    Args:
        record_id (str): Unique identifier for a single record in the training
            dataset, corresponding to the portion of its html / gold standard
            filename before the file extension.
        root_dir (str): Directory on disk containing sub-directories for
            training data: raw html files, gold standard text files, and
            gold standard blocks files.
        min_frac_tokens (float): Minimum fraction of block tokens found
            in gold-standard text extraction for the block to be considered "content".
            Value must be in [0.0, 1.0].

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
        :func:`prepare_all_training_data()`
    """
    if not 0.0 <= min_frac_tokens <= 1.0:
        raise ValueError("`min_frac_tokens` must be in [0.0, 1.0]")

    record = Record(record_id, root_dir=root_dir)
    html = load_html_data(record.html_fpath)
    blocks = load_blocks_data(record.blocks_fpath)
    content_blocks = []
    comments_blocks = []
    for block in blocks:
        frac_tok_in_text, frac_tok_in_comment, block_toks, toks_in_text, toks_in_comment = block.split("\t")
        # total number of tokens in block is used as weights
        num_block_toks = len(block_tokens.split())
        content_blocks.append(
            (float(frac_tok_in_text), num_block_toks, toks_in_text.split())
        )
        comments_blocks.append(
            (float(frac_tok_in_comment), num_block_toks, toks_in_comment.split())
        )
    parsed_content_blocks = _parse_blocks(content_blocks, min_frac_tokens)
    parsed_comments_blocks = _parse_blocks(comments_blocks, min_frac_tokens)
    return (html, parsed_content_blocks, parsed_comments_blocks)


def _parse_blocks(blocks, min_frac_tokens):
    """
    Args:
        blocks (List[str])
        min_frac_tokens (float)

    Returns:
        Tuple[np.ndarray[int], np.ndarray[int], List[str]]
    """
    is_above_thresh = (np.array([block[0] for block in blocks]) > min_frac_tokens).astype(np.int)
    token_counts = np.array([block[1] for block in blocks])
    all_tokens = list(itertools.chain.from_iterable(
        block[2] for block in blocks if block[1] > 0))
    return (is_above_thresh, token_counts, all_tokens)


def load_html_data(fpath):
    """
    Load HTML data for a given record from ``fpath``.

    Args:
        fpath (str)

    Returns:
        str
    """
    html = None
    for encoding in ("utf-8", "iso-8859-1"):  # "utf-16"
        try:
            with io.open(fpath, mode="rt", encoding=encoding) as f:
                html = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            # let's try another encoding...
            pass
    if html:
        html = ftfy.fix_encoding(html).strip()
    return html


def load_text_data(fpath, cleaneval=False):
    """
    Load gold standard text data for a given record from ``fpath``,
    removing tags and/or comments as needed.

    Args:
        fpath (str)
        cleaneval (bool): If True, assume no comments and parse the gold standard
            to remove tags.

    Returns:
        str
    """
    text = None
    for encoding in ("utf-8", "utf-16", "iso-8859-1"):
        try:
            with io.open(fpath, mode="rt", encoding=encoding) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            # let's try another encoding...
            pass
    if not text:
        return None

    if not cleaneval:
        # split out comments, if present as in the old dataset
        if COMMENTS_DELIM in text:
            text, _ = RE_COMMENTS_DELIM.split(text, maxsplit=1)
    else:
        tree = etree.fromstring(text, parser=etree.HTMLParser())
        text = u" ".join(text_from_subtree(tree))
    if text:
        text = ftfy.fix_encoding(text).strip()
    return text


def load_blocks_data(fpath):
    """
    Load gold standard blocks data for a given record from ``fpath``,
    splitting on newlines and filtering out empty lines.

    Args:
        fpath (str)

    Returns:
        List[str]
    """
    with io.open(fpath, mode="rt", encoding="utf-8") as f:
        blocks = f.read()
    return [block for block in blocks.split("\n") if block]


#
# TODO: remove everything below here, when we're ready
#

RAW_HTML_DIRNAME = 'HTML'
GOLD_STANDARD_DIRNAME = 'Corrected'
GOLD_STANDARD_BLOCKS_DIRNAME = 'block_corrected'

RAW_HTML_EXT = '.html'
GOLD_STANDARD_EXT = '.html.corrected.txt'
GOLD_STANDARD_BLOCKS_EXT = '.block_corrected.txt'


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


def get_filenames(dirpath, full_path=False, match_regex=None, extension=None):
    """
    Get all filenames under ``dirpath`` that match ``match_regex`` or have file
    extension equal to ``extension``, optionally prepending the full path.

    Args:
        dirpath (str): /path/to/dir on disk where files to read are stored.
        full_path (bool): If False, return filenames without path; if True,
            return filenames with path, as ``os.path.join(dirpath, fname)``.
        match_regex (str or :class:`re.Pattern`): If specified, only filenames
            matching this regex pattern are included. For example, "^\d+\.html".
        extension (str): If specified, only filenames of a certain type are included.
            For example, ".html".

    Yields:
        str: Next (matching) filename in ``dirpath``.
    """
    if not os.path.isdir(dirpath):
        raise OSError('directory "{}" does not exist'.format(dirpath))
    if match_regex and not isinstance(match_regex, re.Pattern):
        match_regex = re.compile(match_regex, flags=re.UNICODE)
    for fname in sorted(os.listdir(dirpath)):
        if extension and not os.path.splitext(fname)[-1] == extension:
            continue
        if match_regex and not match_regex.search(fname):
            continue
        if full_path is True:
            yield os.path.join(dirpath, fname)
        else:
            yield fname


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
