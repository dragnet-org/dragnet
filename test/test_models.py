import io
import json
import os

import pytest

from dragnet import extract_content, extract_comments, extract_content_and_comments
from dragnet.blocks import simple_tokenizer
from dragnet.util import evaluation_metrics

FIXTURES = os.path.join('test', 'datafiles')


@pytest.fixture(scope="module")
def html():
    with io.open(os.path.join(FIXTURES, "models_testing.html"), mode="rt") as f:
        html_ = f.read()
    return html_


def test_models(html):
    models = [extract_content, extract_comments]  # extract_content_and_comments]

    with io.open(os.path.join(FIXTURES, 'models_content_mod.json'), 'r') as f:
        actual_content = json.load(f)

    for i, model in enumerate(models):
        gold_standard = actual_content[i]
        passed = False
        for i in range(10):
            content = model(html)
            _, _, f1 = evaluation_metrics(
                simple_tokenizer(gold_standard), simple_tokenizer(content))
            if f1 >= 0.8:
                passed = True
                break
        assert passed


def test_content_and_content_comments_extractor(html):
    content = extract_content(html)
    content_comments = extract_comments(html)
    passed_content = False
    passed_content_comments = False
    for i in range(10):
        # actual_content, actual_content_comments = \
        #     extract_content_and_comments(html)
        actual_content = extract_content(html)
        actual_content_comments = extract_comments(html)
        passed_content = actual_content == content
        passed_content_comments = (
            actual_content_comments == content_comments)
        if passed_content and passed_content_comments:
            break

    assert passed_content
    assert passed_content_comments


def test_content_and_content_comments_extractor_blocks(html):
    """
    The content and content/comments extractor should return proper blocks
    """
    content = extract_content(html, as_blocks=True)
    content_comments = extract_comments(html, as_blocks=True)

    passed_content = False
    passed_content_comments = False
    for i in range(5):
        # actual_content, actual_content_comments = \
        #     content_and_content_comments_extractor.analyze(
        #         html, blocks=True)
        actual_content = extract_content(html, as_blocks=True)
        actual_content_comments = extract_comments(html, as_blocks=True)
        passed_content = (
            [blk.text for blk in actual_content] ==
            [blk.text for blk in content]
        )
        passed_content_comments = (
            [blk.text for blk in actual_content_comments] ==
            [blk.text for blk in content_comments]
        )
        if passed_content and passed_content_comments:
            break

    assert passed_content
    assert passed_content_comments
