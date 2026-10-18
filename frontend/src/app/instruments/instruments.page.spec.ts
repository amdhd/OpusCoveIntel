/**
 * The instrument catalogue.
 *
 * One thing worth pinning: selecting an instrument is done by a control, not
 * by a row. A `(click)` on a `<tr>` is unreachable by keyboard and invisible
 * to a screen reader, and this table is how a covenant gets opened.
 */
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { Api } from '../api/api';
import type { CovenantRead, InstrumentRead } from '../api/models';
import { InstrumentsPage } from './instruments.page';

function instrument(overrides: Partial<InstrumentRead> = {}): InstrumentRead {
  return {
    id: 'instrument-1',
    instrument_name: 'RM500m Wakalah Sukuk',
    issuer_name: 'Synthetic Infrastructure Holdings Berhad',
    currency: 'MYR',
    issue_size: '500000000.0000',
    sukuk_structure: 'wakalah',
    current_rating: 'AA3',
    current_rating_notch: 'AA-',
    rating_agency: 'RAM',
    maturity_date: '2031-12-01',
    ...overrides,
  } as InstrumentRead;
}

function createApiMock() {
  return {
    listInstruments: vi.fn<Api['listInstruments']>(),
    instrumentCovenants: vi.fn<Api['instrumentCovenants']>(),
  };
}

describe('InstrumentsPage', () => {
  let api: ReturnType<typeof createApiMock>;

  beforeEach(() => {
    api = createApiMock();
    api.listInstruments.mockReturnValue(of([instrument()]));
    api.instrumentCovenants.mockReturnValue(of([] as CovenantRead[]));

    TestBed.configureTestingModule({
      providers: [provideRouter([]), { provide: Api, useValue: api }],
    });
  });

  it('opens an instrument from a real control, not from the row', () => {
    const fixture = TestBed.createComponent(InstrumentsPage);
    fixture.detectChanges();

    const row: HTMLElement = fixture.nativeElement.querySelector('tbody tr');
    const control: HTMLButtonElement | null = row.querySelector('button');

    // A keyboard reaches a button. It does not reach a `<tr>`.
    expect(control).not.toBeNull();
    expect(control!.textContent?.trim()).toBe('RM500m Wakalah Sukuk');
    expect(control!.getAttribute('aria-pressed')).toBe('false');

    control!.click();
    fixture.detectChanges();

    expect(api.instrumentCovenants).toHaveBeenCalledWith('instrument-1');
    expect(control!.getAttribute('aria-pressed')).toBe('true');
    expect(row.classList).toContain('selected');
  });
});
