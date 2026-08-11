/**
 * Upload a document, and watch it being ingested.
 *
 * This screen is the fix for finding 7: the UI had four screens and none of
 * them could get a document into the system, so the only way in was the CLI or
 * a raw `POST /documents/upload`.
 *
 * Two progress phases, because they fail differently and a single bar would
 * lie about both:
 *
 *  1. **Transfer** -- bytes leaving the browser, reported by `HttpClient`. A
 *     500-page prospectus is tens of megabytes; a screen with no feedback
 *     during that is indistinguishable from one that has hung.
 *  2. **Ingestion** -- the worker parsing, scoring pages and chunking, polled
 *     from `GET /documents/{id}/status`. This is the part that can take a
 *     minute, and the part that can fail with a reason worth reading.
 *
 * Polling stops on the server's `terminal` flag, never on a status this file
 * recognises. A client that decided for itself when ingestion was finished
 * would report a half-parsed document as done the first time a status was
 * added to the pipeline.
 */
import { HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { Component, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, exhaustMap, takeWhile, timer } from 'rxjs';

import { Api } from '../api/api';
import { ConfidencePipe, LabelPipe, label } from '../format/format';
import { FAILED_STATUSES } from '../api/models';
import type { DocumentRead, DocumentStatusRead, DocumentType } from '../api/models';

/**
 * The types the upload form offers.
 *
 * `satisfies` rejects a member the API does not accept -- this list used to
 * offer `financial_statement`, which is not a `DocumentType` and which the
 * endpoint answers with a 422. `Missing` is the other direction: a document
 * type added to the enum and not offered here fails the build rather than
 * quietly becoming unreachable from the only screen that can set one.
 */
const DOCUMENT_TYPES = [
  'unknown',
  'prospectus',
  'information_memorandum',
  'trust_deed',
  'rating_report',
  'announcement',
  'supplemental',
] as const satisfies readonly DocumentType[];

type Missing = Exclude<DocumentType, (typeof DOCUMENT_TYPES)[number]>;
const _EVERY_TYPE_IS_OFFERED: Missing extends never ? true : never = true;

/** How often to ask the server where a document has got to. */
const POLL_MS = 1500;

/** The worker polls every 5s, so a document can sit queued for a while. */
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

@Component({
  selector: 'app-documents',
  imports: [FormsModule, ConfidencePipe, LabelPipe],
  templateUrl: './documents.page.html',
})
export class DocumentsPage implements OnDestroy {
  private readonly api = inject(Api);
  private polling?: Subscription;

  protected readonly documents = signal<DocumentRead[]>([]);
  protected readonly loading = signal(true);

  protected file: File | null = null;
  protected documentType: DocumentType = 'unknown';

  /** 0-100 while bytes are in flight, null when nothing is uploading. */
  protected readonly transferPercent = signal<number | null>(null);
  protected readonly watching = signal<DocumentStatusRead | null>(null);
  protected readonly notice = signal<string | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly dragging = signal(false);

  constructor() {
    this.refresh();
  }

  ngOnDestroy(): void {
    this.polling?.unsubscribe();
  }

  protected readonly documentTypes = DOCUMENT_TYPES;

  // -- picking a file ----------------------------------------------------

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.take(input.files?.[0] ?? null);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    this.take(event.dataTransfer?.files?.[0] ?? null);
  }

  protected onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  protected onDragLeave(): void {
    this.dragging.set(false);
  }

  /**
   * Refuse a non-PDF here as well as at the server.
   *
   * The server is the authority and rejects it with a 415 either way; this
   * only saves someone uploading 40MB to be told afterwards.
   */
  private take(file: File | null): void {
    this.error.set(null);
    this.notice.set(null);
    if (file && !file.name.toLowerCase().endsWith('.pdf')) {
      this.error.set('Only PDFs can be ingested.');
      this.file = null;
      return;
    }
    this.file = file;
  }

  // -- uploading ---------------------------------------------------------

  protected upload(): void {
    if (!this.file) {
      return;
    }
    this.error.set(null);
    this.notice.set(null);
    this.watching.set(null);
    this.transferPercent.set(0);

    this.api.uploadDocument(this.file, this.documentType).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.transferPercent.set(Math.round((100 * event.loaded) / event.total));
        }
        if (event.type === HttpEventType.Response && event.body) {
          this.transferPercent.set(null);
          this.file = null;
          // A duplicate is a correct outcome, not an error: the same bytes are
          // one document. Say so plainly rather than showing a failure.
          this.notice.set(
            event.body.duplicate
              ? `Already ingested as ${event.body.document.filename}; showing the existing document.`
              : `Uploaded ${event.body.document.filename}.`,
          );
          this.refresh();
          this.watch(event.body.document.id);
        }
      },
      error: (caught: unknown) => {
        this.transferPercent.set(null);
        this.error.set(this.describe(caught));
      },
    });
  }

  // -- watching ----------------------------------------------------------

  /** Poll a document's status until the server says nothing more will happen. */
  protected watch(documentId: string): void {
    this.polling?.unsubscribe();
    const startedAt = Date.now();

    this.polling = timer(0, POLL_MS)
      .pipe(
        // `exhaustMap`, not `switchMap`: a slow status request must not have a
        // second one stacked behind it on every tick.
        exhaustMap(() => this.api.documentStatus(documentId)),
        // Inclusive, so the terminal status is delivered and *then* the stream
        // completes. The server decides when to stop, not this file.
        takeWhile((status) => !status.terminal, true),
      )
      .subscribe({
        next: (status) => {
          this.watching.set(status);
          if (!status.terminal && Date.now() - startedAt > POLL_TIMEOUT_MS) {
            // Stop watching, but do not claim it failed -- the worker may be
            // down, and the document is still safely queued.
            this.polling?.unsubscribe();
            this.error.set(
              'Still queued after five minutes. The worker may not be running; ' +
                'the document is safe and will be picked up when it is.',
            );
          }
        },
        error: (caught: unknown) => this.error.set(this.describe(caught)),
        // Only reached by the terminal status above, never by the timeout.
        complete: () => this.refresh(),
      });
  }

  /** Parse now rather than waiting for the worker -- for demos and operators. */
  protected processNow(documentId: string): void {
    this.error.set(null);
    this.api.processDocument(documentId).subscribe({
      next: () => this.watch(documentId),
      error: (caught: unknown) => this.error.set(this.describe(caught)),
    });
  }

  protected refresh(): void {
    this.loading.set(true);
    this.api.listDocuments().subscribe({
      next: (documents) => {
        this.documents.set(documents);
        this.loading.set(false);
      },
      error: (caught: unknown) => {
        this.error.set(this.describe(caught));
        this.loading.set(false);
      },
    });
  }

  // -- presentation ------------------------------------------------------

  protected failed(status: string): boolean {
    return FAILED_STATUSES.has(status);
  }

  /**
   * What to print in the `type` column.
   *
   * `unknown` is the default the upload endpoint applies when nobody said
   * otherwise -- the absence of a classification rather than a kind of
   * document. Printed as the enum it reads as a finding about the file.
   * Underscores go the same way the server-rendered templates already send
   * them, so `trust_deed` reads as a phrase on both.
   */
  protected typeLabel(documentType: string): string {
    return documentType === 'unknown' ? 'Not classified' : label(documentType);
  }

  protected statusClass(status: string): string {
    if (this.failed(status)) {
      return 'breach';
    }
    return status === 'uploaded' || status === 'parsing' ? 'at_risk' : 'ok';
  }

  private describe(caught: unknown): string {
    if (!(caught instanceof HttpErrorResponse)) {
      return 'Something went wrong.';
    }
    // The server's message is the useful one -- "upload exceeds the 100 MB
    // limit" beats "413".
    const detail = (caught.error as { detail?: string } | null)?.detail;
    if (detail) {
      return detail;
    }
    return caught.status === 0 ? 'Cannot reach the server.' : `Request failed (${caught.status}).`;
  }
}
