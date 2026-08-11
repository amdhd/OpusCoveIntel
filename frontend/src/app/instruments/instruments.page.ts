/**
 * The instrument catalogue, and one instrument's covenants.
 *
 * A covenant the machine was unsure about must not look like one it was sure
 * about: anything under the review threshold is marked, and the provenance
 * link leaves for the server-rendered viewer, which is still where a quote is
 * checked against its chunk.
 */
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { Api } from '../api/api';
import type { CovenantRead, InstrumentRead } from '../api/models';
import { MoneyPipe } from '../format/format';

/** CLAUDE.md 5: below this a field is queued for a human. */
const REVIEW_THRESHOLD = 0.85;

@Component({
  selector: 'app-instruments',
  imports: [FormsModule, MoneyPipe],
  templateUrl: './instruments.page.html',
})
export class InstrumentsPage {
  private readonly api = inject(Api);

  protected readonly instruments = signal<InstrumentRead[]>([]);
  protected readonly selected = signal<InstrumentRead | null>(null);
  protected readonly covenants = signal<CovenantRead[]>([]);
  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);

  constructor() {
    this.api.listInstruments().subscribe({
      next: (instruments) => {
        this.instruments.set(instruments);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Could not load instruments.');
        this.loading.set(false);
      },
    });
  }

  protected select(instrument: InstrumentRead): void {
    this.selected.set(instrument);
    this.covenants.set([]);
    this.api.instrumentCovenants(instrument.id).subscribe({
      next: (covenants) => this.covenants.set(covenants),
      error: () => this.error.set('Could not load covenants for that instrument.'),
    });
  }

  protected lowConfidence(covenant: CovenantRead): boolean {
    // `!= null` rather than `!== null`: the generated type is optional as
    // well as nullable, so a missing field is `undefined`, not `null`.
    return covenant.confidence != null && covenant.confidence < REVIEW_THRESHOLD;
  }
}
