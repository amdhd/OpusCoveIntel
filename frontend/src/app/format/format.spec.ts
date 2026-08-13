/**
 * Display formatting.
 *
 * `CASES` is duplicated verbatim from `tests/test_web_format.py`. That
 * duplication is the point: the two renderers show the same portfolio, and the
 * only way to notice they have drifted is to assert the same table on both
 * sides.
 */
import {
  belowReviewBar,
  confidence,
  EM_DASH,
  label,
  money,
  REVIEW_THRESHOLD,
  timestamp,
} from './format';

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

describe('confidence', () => {
  // Same table as `TestConfidence` in tests/test_web_format.py.
  const CONFIDENCE_CASES: [number, string][] = [
    [0.78, '78%'],
    [0.85, '85%'],
    [0.9999, '100%'],
    [0.0, '0%'],
    [1.0, '100%'],
  ];

  for (const [value, expected] of CONFIDENCE_CASES) {
    it(`renders ${value} as ${expected}`, () => {
      expect(confidence(value)).toBe(expected);
    });
  }

  it('renders nothing as a dash', () => {
    expect(confidence(null)).toBe(EM_DASH);
    expect(confidence(undefined)).toBe(EM_DASH);
  });

  it('marks a figure under the bar, and only under it', () => {
    expect(belowReviewBar(REVIEW_THRESHOLD - 0.01)).toBeTrue();
    // The bar itself is not below itself: `confidence < threshold` queues an
    // item, so 0.85 is settled and must not wear the marker.
    expect(belowReviewBar(REVIEW_THRESHOLD)).toBeFalse();
    expect(belowReviewBar(null)).toBeFalse();
    expect(belowReviewBar(undefined)).toBeFalse();
  });
});

describe('label', () => {
  // Same table as `LABEL_CASES` in tests/test_web_format.py.
  const LABEL_CASES: [string, string][] = [
    ['financial_covenant', 'Financial covenant'],
    ['rating_report', 'Rating report'],
    ['at_risk', 'At risk'],
    ['not_applicable', 'Not applicable'],
    // Sentence case, not title case: 'Rating report', never 'Rating Report'.
    ['trust_deed', 'Trust deed'],
    // An initialism stays one. 'Llm' reads as a typo.
    ['llm', 'LLM'],
    ['vlm', 'VLM'],
    ['spv', 'SPV'],
    // A word the data already capitalised keeps its shape: sukuk structures are
    // proper nouns (CLAUDE.md 6) and must not be flattened.
    ['Ijarah', 'Ijarah'],
    ['wakalah', 'Wakalah'],
  ];

  for (const [value, expected] of LABEL_CASES) {
    it(`renders ${JSON.stringify(value)} as ${JSON.stringify(expected)}`, () => {
      expect(label(value)).toBe(expected);
    });
  }

  for (const empty of [null, undefined, '', '   ']) {
    it(`renders ${JSON.stringify(empty)} as a dash`, () => {
      expect(label(empty)).toBe(EM_DASH);
    });
  }
});

describe('timestamp', () => {
  // The defect this landed for: a raw TIMESTAMPTZ off the status endpoint.
  it('reads a stored instant as a person would, in UTC', () => {
    expect(timestamp('2026-08-07T09:06:01.509898Z')).toBe('7 Aug 2026, 09:06:01 UTC');
  });

  it('does not shift with microseconds or a missing fraction', () => {
    expect(timestamp('2026-12-25T00:00:00Z')).toBe('25 Dec 2026, 00:00:00 UTC');
  });

  it('leaves something it does not recognise alone rather than swallowing it', () => {
    expect(timestamp('not a date')).toBe('not a date');
  });

  for (const empty of [null, undefined, '']) {
    it(`renders ${JSON.stringify(empty)} as a dash`, () => {
      expect(timestamp(empty)).toBe(EM_DASH);
    });
  }
});
