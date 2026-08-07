/* Progressive enhancement only.
 *
 * Every action on these pages is a plain form POST that works with this file
 * absent. Nothing here is load-bearing, and nothing here talks to the API --
 * a UI that quietly depends on script is a UI that breaks in ways nobody can
 * reproduce.
 *
 * No framework, and no vendored third-party bundle: four screens do not earn
 * one, and there is nothing here that warrants shipping code we did not read.
 */

(function () {
  "use strict";

  /* Disable a submit button after the first click. The agent takes a couple of
   * seconds to answer and the review actions are not idempotent -- a second
   * click on "Approve" earns a 409, which is correct but looks like a fault. */
  document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector("button[type=submit]");
      if (!button) return;
      // Deferred so the button's value still makes it into the submission.
      window.setTimeout(function () {
        button.disabled = true;
        if (button.dataset.busy) button.textContent = button.dataset.busy;
      }, 0);
    });
  });

  /* Bring the highlighted quote into view on the provenance page.
   *
   * Scrolls the chunk pane, never the document. `scrollIntoView` did the
   * latter: on a long clause it scrolled the whole page to centre the mark,
   * pushing the nav bar and the "Source" heading off the top, so the page
   * looked broken on arrival. Only the inner pane should move. */
  var mark = document.querySelector(".chunk mark");
  var chunk = mark && mark.closest(".chunk");
  if (mark && chunk && mark.offsetTop > chunk.clientHeight) {
    chunk.scrollTop = mark.offsetTop - chunk.clientHeight / 3;
  }

  /* Ctrl/Cmd+Enter submits the question, since it lives in a textarea and
   * Enter has to stay available for line breaks. */
  var question = document.getElementById("question");
  if (question) {
    question.addEventListener("keydown", function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        question.form.requestSubmit();
      }
    });
  }
})();
