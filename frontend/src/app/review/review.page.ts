/**
 * The review queue: the one screen where a human overrides the machine.
 *
 * Only a reviewer may decide an item. The buttons are hidden for an analyst,
 * but the decision is authorised on the server (403 from `require_reviewer`) --
 * hiding a control is courtesy, not a permission check.
 *
 * Who decided is taken from the session, never from this client. That is the
 * whole reason `users` exists: a reviewer id supplied by the caller is a claim,
 * not a record.
 */
import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { Api } from '../api/api';
import type { ReviewItem } from '../api/models';
import { belowReviewBar, ConfidencePipe, MoneyPipe, REVIEW_THRESHOLD } from '../format/format';
import { Auth } from '../auth/auth';

@Component({
  selector: 'app-review',
  imports: [FormsModule, ConfidencePipe, MoneyPipe],
  templateUrl: './review.page.html',
})
export class ReviewPage {
  private readonly api = inject(Api);
  protected readonly auth = inject(Auth);

  protected readonly items = signal<ReviewItem[]>([]);
  protected readonly totalPending = signal(0);
  protected readonly loading = signal(true);
  protected readonly busy = signal<string | null>(null);
  protected readonly notice = signal<string | null>(null);
  protected readonly error = signal<string | null>(null);

  protected readonly belowBar = belowReviewBar;
  /** For the marker's tooltip, so the bar is named rather than remembered. */
  protected readonly reviewBar = `${Math.round(REVIEW_THRESHOLD * 100)}%`;

  /** Correction drafts, keyed by review id, so two open rows cannot collide. */
  protected corrections: Record<string, string> = {};

  constructor() {
    this.refresh();
  }

  protected refresh(): void {
    this.loading.set(true);
    this.api.pendingReviews().subscribe({
      next: (response) => {
        this.items.set(response.items);
        this.totalPending.set(response.total_pending);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Could not load the review queue.');
        this.loading.set(false);
      },
    });
  }

  protected approve(item: ReviewItem): void {
    this.act(item.id, this.api.approveReview(item.id), 'approved');
  }

  protected reject(item: ReviewItem): void {
    this.act(item.id, this.api.rejectReview(item.id), 'rejected');
  }

  protected correct(item: ReviewItem): void {
    const value = (this.corrections[item.id] ?? '').trim();
    if (!value) {
      this.error.set('A correction needs a value.');
      return;
    }
    this.act(item.id, this.api.correctReview(item.id, value), 'corrected');
  }

  private act(reviewId: string, request: ReturnType<Api['approveReview']>, verb: string): void {
    this.busy.set(reviewId);
    this.error.set(null);
    this.notice.set(null);
    request.subscribe({
      next: () => {
        this.busy.set(null);
        this.notice.set(`Item ${verb}.`);
        delete this.corrections[reviewId];
        this.refresh();
      },
      error: (caught: unknown) => {
        this.busy.set(null);
        if (caught instanceof HttpErrorResponse && caught.status === 409) {
          // Someone else decided it first. Reload rather than insist.
          this.error.set('That item was already decided. Refreshing the queue.');
          this.refresh();
          return;
        }
        if (caught instanceof HttpErrorResponse && caught.status === 403) {
          this.error.set('Your role may not decide review items.');
          return;
        }
        this.error.set('The decision did not go through.');
      },
    });
  }
}
