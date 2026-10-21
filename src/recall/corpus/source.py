"""Where the corpus comes from: FastAPI's English documentation at a pinned commit, and which pages Recall keeps.

DESIGN.md "The corpus" gives the reasons for each exclusion.
"""

from __future__ import annotations

import posixpath

REPOSITORY = "fastapi/fastapi"
TAG = "0.157.0"
COMMIT = "386a28af2f87af1a32ac26d88f07b7f58ab2dfb1"
LICENSE = "MIT"
ARCHIVE_URL = f"https://github.com/{REPOSITORY}/archive/{COMMIT}.tar.gz"

PAGES_ROOT = "docs/en/docs/"
LICENSE_PATH = "LICENSE"
SITE_URL = "https://fastapi.tiangolo.com/"

EXCLUDED_DIRS = {
    "reference/": "generated API reference: mkdocstrings directives with almost no text",
}
EXCLUDED_PAGES = {
    "release-notes.md": "changelog: half the corpus's bytes, naming versions rather than behaviour",
    "_llm-test.md": "an input to the translation tooling",
    "translation-banner.md": "an input to the translation tooling",
    "fastapi-people.md": "a generated list of people",
    "external-links.md": "a generated list of links",
    "management.md": "project administration",
    "newsletter.md": "a sign-up page",
}


def page_of(repo_path: str) -> str | None:
    """The page path ('tutorial/query-params.md') of a kept page's repository path, or None if it isn't one."""
    if not repo_path.startswith(PAGES_ROOT) or not repo_path.endswith(".md"):
        return None
    page = repo_path.removeprefix(PAGES_ROOT)
    if page in EXCLUDED_PAGES or any(page.startswith(d) for d in EXCLUDED_DIRS):
        return None
    return page


def repo_path_of(page: str) -> str:
    return PAGES_ROOT + page


def page_url(page: str) -> str:
    """The page's address on the published site (mkdocs directory URLs: index.md is its directory)."""
    stem = page.removesuffix(".md")
    if stem == "index":
        return SITE_URL
    if posixpath.basename(stem) == "index":
        return SITE_URL + posixpath.dirname(stem) + "/"
    return SITE_URL + stem + "/"
