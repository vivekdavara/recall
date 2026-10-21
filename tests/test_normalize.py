from __future__ import annotations

from pathlib import Path

import pytest

from recall.corpus.normalize import NormalizeError, include_path, includes, normalize, title

FILES = {
    "docs_src/q/tutorial001.py": (
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/')\ndef root():\n    return 1\n"
    ),
    "docs_src/t/item.html": "<html>\n<b>{{ id }}</b>\n</html>\n",
    "docs_src/q/fenced.py": 'doc = """\n```\nnot a fence closer here\n```\n"""\n',
    "docs_src/q/crlf.py": "a = 1\r\nb = 2\r\n",
}


def read(path: str) -> str:
    try:
        return FILES[path]
    except KeyError:
        raise FileNotFoundError(path) from None


def norm(text: str) -> str:
    return normalize("tutorial/page.md", text, read)


def test_a_code_include_becomes_a_python_block():
    out = norm("Intro.\n\n{* ../../docs_src/q/tutorial001.py hl[6] *}\n\nAfter.")
    assert out == (
        "Intro.\n\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/')\ndef root():\n"
        "    return 1\n```\n\nAfter."
    )


def test_ln_selects_lines_and_marks_what_is_left_out():
    out = norm("{* ../../docs_src/q/tutorial001.py ln[1,3:3,6:7] hl[2] *}")
    # Line 2 (blank) and lines 4-5 are left out.
    assert out == "```python\nfrom fastapi import FastAPI\n...\napp = FastAPI()\n...\n@app.get('/')\ndef root():\n```"
    assert norm("{* ../../docs_src/q/tutorial001.py ln[1:2,3] *}").count("...") == 0


@pytest.mark.parametrize("spec", ["0:2", "2:1", "1:99", "3,1", "1:x"])
def test_bad_line_ranges_are_errors(spec: str):
    with pytest.raises(NormalizeError):
        norm(f"{{* ../../docs_src/q/tutorial001.py ln[{spec}] *}}")


def test_a_code_include_in_a_list_keeps_its_indent():
    out = norm("* step\n\n    {* ../../docs_src/q/crlf.py *}\n")
    assert out == "* step\n\n    ```python\n    a = 1\n    b = 2\n    ```\n"


def test_a_code_include_whose_code_has_a_fence_gets_a_longer_fence():
    out = norm("{* ../../docs_src/q/fenced.py *}")
    assert out.startswith("````python\n") and out.endswith("\n````")


def test_a_code_include_inside_a_fence_is_left_alone():
    page = "```python\nnew = {**old}\n{* ../../docs_src/q/tutorial001.py *}\n```"
    assert norm(page) == page


def test_a_raw_include_inside_a_fence_is_expanded_verbatim():
    out = norm('```jinja hl_lines="2"\n{!../../docs_src/t/item.html!}\n```')
    assert out == '```jinja hl_lines="2"\n<html>\n<b>{{ id }}</b>\n</html>\n```'
    assert norm("```\n{!> ../../docs_src/t/item.html!}\n```") == "```\n<html>\n<b>{{ id }}</b>\n</html>\n```"


def test_a_raw_include_that_could_close_its_fence_is_refused():
    with pytest.raises(NormalizeError, match="close its fence"):
        norm("```\n{!../../docs_src/q/fenced.py!}\n```")


def test_a_missing_include_is_an_error_naming_the_page_and_the_file():
    with pytest.raises(NormalizeError, match=r"tutorial/page.md: included file not found: docs_src/nope.py"):
        norm("{* ../../docs_src/nope.py *}")


@pytest.mark.parametrize("target", ["../../../etc/passwd", "../../../../x.py"])
def test_an_include_outside_the_repository_is_refused(target: str):
    with pytest.raises(NormalizeError, match="outside the repository"):
        include_path(target)


def test_include_paths_are_relative_to_docs_en():
    assert include_path("../../docs_src/q/a.py") == "docs_src/q/a.py"
    assert include_path("../../fastapi/openapi/docs.py") == "fastapi/openapi/docs.py"


def test_includes_lists_each_file_once_in_order():
    page = (
        "{* ../../docs_src/q/tutorial001.py *}\n```\n{!../../docs_src/t/item.html!}\n```\n"
        "{* ../../docs_src/q/tutorial001.py ln[1] *}\n```\n{* ../../docs_src/q/ignored.py *}\n```"
    )
    assert includes("p.md", page) == ["docs_src/q/tutorial001.py", "docs_src/t/item.html"]


def test_admonitions_become_label_lines_and_their_closers_blank_lines():
    page = "/// tip\n\nUse it.\n\n///\n/// note | Technical Details\nMore.\n///\nNext."
    assert norm(page) == "Tip:\n\nUse it.\n\n\nNote: Technical Details\nMore.\n\nNext."


def test_tabs_become_their_label():
    assert norm("//// tab | Python 3.10+\n\ncode\n\n////") == "Python 3.10+:\n\ncode\n\n"


def test_a_marker_inside_a_fence_is_code():
    page = "```\n/// tip\n```"
    assert norm(page) == page


def test_html_tags_go_and_their_text_stays():
    page = 'Data <dfn title="converting the string">"parsing"</dfn> and <abbr title="x">LLM</abbr>.<br/>'
    assert norm(page) == 'Data "parsing" and LLM.'


def test_tags_inside_inline_code_and_fences_stay():
    assert norm("Use `<b>` and ``<i> `x` </i>``, not <b>this</b>.") == "Use `<b>` and ``<i> `x` </i>``, not this."
    page = "```html\n<b>bold</b>\n```"
    assert norm(page) == page


def test_html_comments_go_even_across_lines():
    page = "a <!-- one --> b <!-- two\nthree\nfour --> c\nd"
    assert norm(page) == "a  b \n\n c\nd"


def test_heading_ids_stay_for_the_chunker():
    assert norm("## Defaults { #defaults }") == "## Defaults { #defaults }"


def test_crlf_becomes_lf_and_nul_is_refused():
    assert norm("a\r\nb") == "a\nb"
    with pytest.raises(NormalizeError, match="NUL"):
        norm("a\x00b")


def test_title_is_the_first_level_one_heading_outside_code():
    assert title("```\n# not this\n```\n## Sub\n# Query Parameters { #q }\n", "x.md") == "Query Parameters"
    assert title("no heading", "x.md") == "x.md"


def test_a_page_with_every_extension_normalizes_to_its_golden_file():
    # tests/golden/page.md is synthetic; page.normalized.md was written by normalize and checked by eye.
    # Regenerate it only for a deliberate change, and read the diff.
    root = Path(__file__).parent / "golden"

    def read_golden(path: str) -> str:
        return (root / path).read_text(encoding="utf-8")

    out = normalize("golden/page.md", read_golden("page.md"), read_golden)
    assert out == read_golden("page.normalized.md")
    assert normalize("golden/page.md", out, read_golden) == out
