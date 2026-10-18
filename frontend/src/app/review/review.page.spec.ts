/**
 * The review queue's decisions.
 *
 * What is worth pinning here is the shape of what reaches the server. Reject
 * carried no reason for as long as the screen existed: `rejectReview(id)`
 * serialised to `{"reason": null}`, and `RejectRequest.reason` is `str` with
 * `min_length=1`, so every rejection was answered 422 and the button had never
 * once worked. Nothing caught it, because nothing asserted the payload.
 */
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { Api } from '../api/api';
import type { PendingResponse, ReviewActionResponse, ReviewItem } from '../api/models';
import { Auth } from '../auth/auth';
import { ReviewPage } from './review.page';

function item(overrides: Partial<ReviewItem> = {}): ReviewItem {
  return {
    id: 'review-1',
    entity_type: 'covenant',
    field_name: 'threshold_amount',
    old_value: '500000000',
    new_value: null,
    trigger_reason: 'low_confidence',
    confidence: 0.78,
    page_number: 42,
    source_quote: 'The Issuer shall maintain a consolidated net worth of RM500,000,000.',
    ...overrides,
  } as ReviewItem;
}

function createApiMock() {
  return {
    pendingReviews: vi.fn<Api['pendingReviews']>(),
    approveReview: vi.fn<Api['approveReview']>(),
    correctReview: vi.fn<Api['correctReview']>(),
    rejectReview: vi.fn<Api['rejectReview']>(),
    me: vi.fn<Api['me']>(),
  };
}

describe('ReviewPage', () => {
  let api: ReturnType<typeof createApiMock>;

  beforeEach(() => {
    api = createApiMock();
    api.pendingReviews.mockReturnValue(
      of({ items: [item()], total_pending: 1 } as PendingResponse),
    );
    api.rejectReview.mockReturnValue(of({} as ReviewActionResponse));
    api.correctReview.mockReturnValue(of({} as ReviewActionResponse));

    TestBed.configureTestingModule({
      providers: [provideRouter([]), { provide: Api, useValue: api }],
    });
    // The buttons render only for a reviewer.
    vi.spyOn(TestBed.inject(Auth), 'canReview').mockReturnValue(true);
  });

  function page(): {
    reject(i: ReviewItem): void;
    rejections: Record<string, string>;
    error(): string | null;
  } {
    const fixture = TestBed.createComponent(ReviewPage);
    fixture.detectChanges();
    return fixture.componentInstance as never;
  }

  it('sends the typed reason with a rejection', () => {
    const component = page();
    component.rejections['review-1'] = '  duplicate of clause 12.3  ';

    component.reject(item());

    // Trimmed, because ` ` satisfies `min_length=1` on the server and satisfies
    // nobody reading the audit trail.
    expect(api.rejectReview).toHaveBeenCalledWith('review-1', 'duplicate of clause 12.3');
  });

  it('refuses to reject without one rather than letting the server 422', () => {
    const component = page();
    component.rejections['review-1'] = '   ';

    component.reject(item());

    expect(api.rejectReview).not.toHaveBeenCalled();
    expect(component.error()).toBe('A rejection needs a reason.');
  });
});
