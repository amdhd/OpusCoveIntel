/**
 * Display formatting.
 *
 * `CASES` is duplicated verbatim from `tests/test_web_format.py`. That
 * duplication is the point: the two renderers show the same portfolio, and the
 * only way to notice they have drifted is to assert the same table on both
 * sides.
 */
import { EM_DASH, money } from './format';

// [value, currency, expected]
const CASES: [string, string | null, string][] = [
  // The defect this landed for: a NUMERIC(20,4) straight off Postgres.
  ['300000000.0000', 'MYR', 'MYR 300,000,000'],
  ['500000000', null, '500,000,000'],
  // Real sen survive; zero sen do not.
  ['1234.5600', null, '1,234.56'],
  ['1234.0000', null, '1,234'],
  ['0.05', 'MYR', 'MYR 0.05'],
  ['0', 'MYR', 'MYR 0'],
  // Under four digits there is nothing to group.
  ['999', null, '999'],
  ['1000', null, '1,000'],
  ['-2500.50', null, '-2,500.5'],
  // Not a bare decimal: a review value someone typed. Left alone.
  ['RM30,000,000 or its equivalent', null, 'RM30,000,000 or its equivalent'],
  ['n/a', null, 'n/a'],
];

describe('money', () => {
  for (const [value, currency, expected] of CASES) {
    it(`renders ${JSON.stringify(value)} as ${JSON.stringify(expected)}`, () => {
      expect(money(value, currency)).toBe(expected);
    });
  }

  it('keeps every digit of an amount a double would round', () => {
    // The whole reason Decimals cross the wire as strings. `Number()` on this
    // gives 300000000.05000001 on some inputs; string surgery cannot.
    expect(money('300000000.05')).toBe('300,000,000.05');
    expect(money('9007199254740993')).toBe('9,007,199,254,740,993');
  });

  for (const empty of [null, undefined, '']) {
    it(`renders ${JSON.stringify(empty)} as a dash`, () => {
      expect(money(empty)).toBe(EM_DASH);
    });
  }
});
