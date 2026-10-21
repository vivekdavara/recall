# Every Extension { #every-extension }

This synthetic page uses each documentation-build extension that normalization handles. 

## Code includes { #code-includes }

A whole file:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Some of its lines, with a title:

```python
from fastapi import FastAPI
...

@app.get("/items/{item_id}")
async def read_item(item_id: int):
```

* In a list:

    ```python
    print("short")
    ```

## Raw includes { #raw-includes }

```jinja hl_lines="2"
<html>
<h1>{{ id }}</h1>
</html>
```

## Admonitions and tabs { #admonitions-and-tabs }

Tip:

A tip with no title.



Note: Technical Details

A note with a title, and `<b>` in code.



Python 3.10+:

```python
x: int | None = None
```



Python 3.9+:

```python
x: Optional[int] = None
```



## HTML { #html }

Data "parsing", an LLM,
and a link.








```console
$ fastapi dev main.py
# not a heading, and <b>not stripped</b>
```


