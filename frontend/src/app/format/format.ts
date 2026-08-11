/**
 * Display formatting for the client app.
 *
 * **This file has a twin: `app/web/format.py`.** The server-rendered screens
 * and this app show the same portfolio, so they format it the same way or the
 * product looks like two products. The convention is written down in the Python
 * module; the case table in `format.spec.ts` is a copy of the one in
 * `tests/test_web_format.py` on purpose. Change one, change both.
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

@Pipe({ name: 'money' })
export class MoneyPipe implements PipeTransform {
  transform(value: string | number | null | undefined, currency?: string | null): string {
    return money(value, currency);
  }
}
