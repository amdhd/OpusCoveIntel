/**
 * Display formatting for the client app.
 *
 * **This file has a twin: `app/web/format.py`.** The server-rendered screens
 * and this app show the same portfolio, so they format it the same way or the
 * product looks like two products. The convention is written down in the Python
 * module; the case table in `format.spec.ts` is a copy of the one in
 * `tests/test_web_format.py` on purpose. Change one, change both.
 *
 * Confidence is a percentage here too: `0.78` is a number an analyst has to
 * convert before they can compare it against the review bar.
 *
 * The rule that is not merely cosmetic: **money is never parsed through a
 * `number`.** `Decimal` crosses the wire as a JSON string (see `HoldingRead`)
 * so that a client cannot silently round RM300,000,000.05 into a double. That
 * only holds if the code that renders it also leaves it as a string, so the
 * grouping below is string surgery -- no `Number()`, no `parseFloat`, no
 * `toLocaleString`.
 */
import { Pipe, PipeTransform } from '@angular/core';

/** What a `Decimal` looks like once it is a JSON string. */
const BARE_DECIMAL = /^(-?)(\d+)(?:\.(\d*))?$/;

export const EM_DASH = '—';

/** Group an exact amount, optionally prefixed with its currency code. */
export function money(value: string | number | null | undefined, currency?: string | null): string {
  if (value === null || value === undefined || value === '') {
    return EM_DASH;
  }

  const match = BARE_DECIMAL.exec(String(value).trim());
  if (match === null) {
    // Free text -- a typed correction, or a value the extractor lifted
    // verbatim. Not ours to reformat.
    return String(value).trim();
  }

  const [, sign, whole, fraction = ''] = match;
  const trimmed = fraction.replace(/0+$/, '');
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const amount = `${sign}${grouped}${trimmed ? `.${trimmed}` : ''}`;

  return currency ? `${currency} ${amount}` : amount;
}

/**
 * CLAUDE.md 5: below this a field is queued for a human.
 *
 * The server reads this from `DEFAULT_CONFIDENCE_THRESHOLD` and hands it to its
 * templates. This app has no endpoint that publishes it, so it is a constant
 * here -- if the setting is ever changed, this is the second place to change.
 */
export const REVIEW_THRESHOLD = 0.85;

/** A model's confidence as a percentage, to the nearest point. */
export function confidence(value: number | null | undefined): string {
  // Not two decimal places: the third significant figure of a model's
  // self-reported confidence is not a real quantity, and printing it invites
  // people to read 0.78 against 0.77 as though the difference meant something.
  return value == null ? EM_DASH : `${Math.round(value * 100)}%`;
}

/** Whether a figure is under the bar, and so not settled. */
export function belowReviewBar(value: number | null | undefined): boolean {
  // `!= null` rather than `!== null`: the generated type is optional as well as
  // nullable, so a missing field is `undefined`, not `null`.
  return value != null && value < REVIEW_THRESHOLD;
}

@Pipe({ name: 'confidence' })
export class ConfidencePipe implements PipeTransform {
  transform(value: number | null | undefined): string {
    return confidence(value);
  }
}

@Pipe({ name: 'money' })
export class MoneyPipe implements PipeTransform {
  transform(value: string | number | null | undefined, currency?: string | null): string {
    return money(value, currency);
  }
}
