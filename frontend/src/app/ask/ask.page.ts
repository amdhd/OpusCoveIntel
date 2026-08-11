/**
 * Ask the query agent.
 *
 * The interesting case is the one that looks like a failure and is not: a
 * refusal. "No supporting evidence in the corpus" is a correct answer
 * (CLAUDE.md 1.5), so it is rendered as an answer with a quieter treatment,
 * never as an error banner. An interface that styled refusals as errors would
 * train analysts to treat the system's honesty as a malfunction.
 */
import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { Api } from '../api/api';
import { ConfidencePipe, LabelPipe } from '../format/format';
import type { QueryResponse } from '../api/models';

@Component({
  selector: 'app-ask',
  imports: [FormsModule, ConfidencePipe, LabelPipe],
  templateUrl: './ask.page.html',
})
export class AskPage {
  private readonly api = inject(Api);

  protected question = '';
  protected readonly answer = signal<QueryResponse | null>(null);
  protected readonly asking = signal(false);
  protected readonly error = signal<string | null>(null);

  protected ask(): void {
    const question = this.question.trim();
    if (!question) {
      return;
    }
    this.asking.set(true);
    this.error.set(null);
    this.api.ask(question).subscribe({
      next: (response) => {
        this.answer.set(response);
        this.asking.set(false);
      },
      error: (caught: unknown) => {
        const detail =
          caught instanceof HttpErrorResponse
            ? ((caught.error as { detail?: string } | null)?.detail ?? `Failed (${caught.status}).`)
            : 'Something went wrong.';
        this.error.set(detail);
        this.asking.set(false);
      },
    });
  }
}
