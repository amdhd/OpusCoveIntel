"""Template content that does not depend on a database.

The ask screen's examples are static markup, so they are asserted against the
rendered template rather than through a request. That keeps the check running
in a checkout with no Postgres, which is where somebody editing a template is
most likely to be.
"""

from __future__ import annotations

import re

from app.evals.golden import GOLDEN_QUESTIONS
from app.web.templates import templates


def _render(name: str, **context: object) -> str:
    return templates.env.get_template(name).render(**context)


class TestTheAskScreenOffersAStartingPoint:
    """A placeholder is guidance that vanishes the moment someone starts
    typing, which is when they need it. The examples stay on the page, and each
    is a submit button carrying its own question, so using one needs no script.
    """

    def test_every_example_is_a_question_the_corpus_can_answer(self) -> None:
        """An example that comes back "no supporting evidence" teaches the
        wrong thing about the product, so they are drawn from the golden set.

        Membership in that set is not enough: a third of it is questions where
        refusing is the correct outcome (G10-G13), and the first version of
        this test happily accepted one of those. `expect_refusal` is the field
        that matters.
        """
        offered = re.findall(r'name="question" value="([^"]+)"', _render("ask.html", user=None))

        assert offered, "the ask screen should offer somewhere to start"
        answerable = {
            question.question for question in GOLDEN_QUESTIONS if not question.expect_refusal
        }
        unevaluated = set(offered) - answerable
        assert not unevaluated, f"examples that are not known-answerable: {sorted(unevaluated)}"

    def test_an_example_asks_without_script(self) -> None:
        """`app/web/static/app.js` is progressive enhancement only, so the
        examples have to work with it absent: each is a submit button whose
        value is the question."""
        html = _render("ask.html", user=None)

        assert 'action="/ui/ask"' in html
        for example in re.findall(r'name="question" value="([^"]+)"', html):
            assert f'<button type="submit" name="question" value="{example}"' in html
