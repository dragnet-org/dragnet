from dragnet.lcs import check_inclusion


def test_check_inclusion():
    inc = check_inclusion(
        ["some", "words", "here", "the", "football"],
        ["he", "said", "words", "kick", "the", "football"])
    assert inc == [False, True, False, True, True]
