from recall import markdown


def fences(text: str) -> list[int | None]:
    return [line.fence for line in markdown.scan(text)]


def test_lines_are_spans_without_their_newline():
    text = "a\n\nbc\n"
    assert [(ln.start, ln.end) for ln in markdown.scan(text)] == [(0, 1), (2, 2), (3, 5), (6, 6)]


def test_a_fence_runs_from_its_opener_to_its_closer():
    assert fences("x\n```py\n# not a heading\n\n```\ny") == [None, 0, 0, 0, 0, None]


def test_a_closer_must_be_as_long_as_the_opener_and_the_same_character():
    text = "````\n```\n~~~~\n````\nafter"
    assert fences(text) == [0, 0, 0, 0, None]


def test_a_closer_may_be_longer_but_carries_nothing_else():
    assert fences("~~~\n~~~ x\n~~~~~\nafter") == [0, 0, 0, None]


def test_an_unclosed_fence_runs_to_the_end():
    assert fences("```\na\n\nb") == [0, 0, 0, 0]


def test_a_backtick_info_string_with_a_backtick_is_not_a_fence():
    assert fences("``` a ` b\ntext") == [None, None]


def test_indented_fences_count_inside_list_items():
    assert fences("* item\n\n    ```python\n    x = 1\n    ```\n") == [None, None, 0, 0, 0, None]


def test_consecutive_fences_get_their_own_index():
    assert fences("```\na\n```\n~~~\nb\n~~~") == [0, 0, 0, 1, 1, 1]


def test_headings_and_their_ids():
    h = markdown.heading("## Defaults { #defaults }")
    assert h == markdown.Heading(level=2, title="Defaults", anchor="defaults")
    assert markdown.heading("# Title #") == markdown.Heading(1, "Title", None)
    assert markdown.heading("### C#") == markdown.Heading(3, "C#", None)
    assert markdown.heading("#") == markdown.Heading(1, "", None)
    assert markdown.heading("   ## three spaces") == markdown.Heading(2, "three spaces", None)


def test_lines_that_are_not_headings():
    for line in ["#hashtag", "    # indented four", "####### seven", "text # not at start", ""]:
        assert markdown.heading(line) is None, line


def test_blank_means_whitespace_only():
    assert markdown.is_blank("") and markdown.is_blank(" \t\N{NO-BREAK SPACE}")
    assert not markdown.is_blank(" x ")
