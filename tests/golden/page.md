# Every Extension { #every-extension }

This synthetic page uses each documentation-build extension that normalization handles. <!-- a comment -->

## Code includes { #code-includes }

A whole file:

{* ../../docs_src/app/main.py hl[6] *}

Some of its lines, with a title:

{* ../../docs_src/app/main.py ln[1,5:7] hl[6] title["main.py"] *}

* In a list:

    {* ../../docs_src/app/short.py *}

## Raw includes { #raw-includes }

```jinja hl_lines="2"
{!../../docs_src/app/item.html!}
```

## Admonitions and tabs { #admonitions-and-tabs }

/// tip

A tip with no title.

///

/// note | Technical Details

A note with a title, and `<b>` in code.

///

//// tab | Python 3.10+

```python
x: int | None = None
```

////

//// tab | Python 3.9+

```python
x: Optional[int] = None
```

////

## HTML { #html }

Data <dfn title="converting the string">"parsing"</dfn>, an <abbr title="Large Language Model">LLM</abbr>,
and <a href="https://example.com" class="external-link" target="_blank">a link</a>.

<!--
A comment
over three lines
-->

<div class="termy">

```console
$ fastapi dev main.py
# not a heading, and <b>not stripped</b>
```

</div>
