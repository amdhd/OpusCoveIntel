/**
 * The upload screen's two hard parts: knowing when ingestion has finished, and
 * not claiming it has when it has not.
 *
 * These drive the component through a mocked `HttpClient`, so they assert what
 * the screen does with the server's answers rather than that Angular renders.
 */
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';

import { HttpErrorResponse } from '@angular/common/http';

import { Api } from '../api/api';
import type { DocumentRead, DocumentStatusRead, UploadResponse } from '../api/models';
import { DocumentsPage } from './documents.page';

function status(overrides: Partial<DocumentStatusRead> = {}): DocumentStatusRead {
  return {
    document_id: 'doc-1',
    filename: 'prospectus.pdf',
    status: 'uploaded',
    page_count: null,
    chunk_count: 0,
    pages_flagged_for_vlm: 0,
    parse_confidence: null,
    jobs: [],
    terminal: false,
    error: null,
    ...overrides,
  } as DocumentStatusRead;
}

describe('DocumentsPage', () => {
  let api: jasmine.SpyObj<Api>;

  beforeEach(() => {
    api = jasmine.createSpyObj<Api>('Api', [
      'listDocuments',
      'uploadDocument',
      'documentStatus',
      'processDocument',
    ]);
    api.listDocuments.and.returnValue(of([]));
    api.documentStatus.and.returnValue(of(status()));

    TestBed.configureTestingModule({
      imports: [DocumentsPage],
      providers: [provideRouter([]), { provide: Api, useValue: api }],
    });
  });

  function create(): DocumentsPage {
    return TestBed.createComponent(DocumentsPage).componentInstance;
  }

  it('does not print an unset classification as a finding about the file', () => {
    const page = create() as unknown as { typeLabel(type: string): string };

    // The upload endpoint defaults to `unknown`, and `opuscovintel ingest`
    // has no way to say otherwise, so this is what most rows hold.
    expect(page.typeLabel('unknown')).toBe('Not classified');
    expect(page.typeLabel('trust_deed')).toBe('Trust deed');
    expect(page.typeLabel('prospectus')).toBe('Prospectus');
  });

  it('says which row the detail panel is describing', fakeAsync(() => {
    // The panel is above the table and every row's button is identical, so
    // without this the screen shows one document's jobs beside five rows of
    // buttons and no way to tell which one you pressed.
    api.listDocuments.and.returnValue(
      of([{ id: 'doc-1', filename: 'prospectus.pdf', document_type: 'unknown' } as DocumentRead]),
    );
    // Terminal, so the poll delivers once and completes rather than leaving a
    // timer running past the end of the test.
    api.documentStatus.and.returnValue(of(status({ terminal: true })));
    const fixture = TestBed.createComponent(DocumentsPage);
    fixture.detectChanges();

    const row = (): HTMLElement => fixture.nativeElement.querySelector('tbody tr');
    const button = (): HTMLButtonElement => row().querySelector('button')!;

    expect(button().textContent?.trim()).toBe('Ingestion detail');
    expect(row().classList).not.toContain('selected');

    button().click();
    tick();
    fixture.detectChanges();

    expect(button().textContent?.trim()).toBe('Showing detail');
    expect(button().getAttribute('aria-pressed')).toBe('true');
    expect(row().classList).toContain('selected');
  }));

  it('reports transfer progress while bytes are in flight', () => {
    api.uploadDocument.and.returnValue(
      of({ type: HttpEventType.UploadProgress, loaded: 512, total: 1024 }),
    );
    const page = create() as unknown as {
      file: File | null;
      upload(): void;
      transferPercent(): number | null;
    };
    page.file = new File([new Uint8Array(1024)], 'p.pdf', { type: 'application/pdf' });

    page.upload();

    expect(page.transferPercent()).toBe(50);
  });

  it('keeps polling while the server says the document is not terminal', fakeAsync(() => {
    const page = create() as unknown as { watch(id: string): void; ngOnDestroy(): void };

    page.watch('doc-1');
    tick(0);
    expect(api.documentStatus).toHaveBeenCalledTimes(1);

    tick(1500);

    expect(api.documentStatus).toHaveBeenCalledTimes(2);
    // Still queued, so the list has not been refreshed as though it were done.
    expect(api.listDocuments).toHaveBeenCalledTimes(1);
    page.ngOnDestroy();
  }));

  it('stops and refreshes the list once the server says terminal', fakeAsync(() => {
    api.documentStatus.and.returnValue(
      of(status({ status: 'chunked', terminal: true, page_count: 4, chunk_count: 12 })),
    );
    const page = create() as unknown as {
      watch(id: string): void;
      watching(): DocumentStatusRead | null;
    };

    page.watch('doc-1');
    tick(0);

    expect(page.watching()?.terminal).toBeTrue();
    // Refreshed exactly once more than the constructor's load.
    expect(api.listDocuments).toHaveBeenCalledTimes(2);

    // And stopped: a poller that keeps asking after the answer is in has not
    // stopped, it is just quiet.
    tick(10_000);
    expect(api.documentStatus).toHaveBeenCalledTimes(1);
  }));

  it('shows the server’s reason rather than a status code', () => {
    api.uploadDocument.and.returnValue(
      throwError(
        () =>
          new HttpErrorResponse({
            status: 413,
            error: { detail: 'upload exceeds the 100 MB limit' },
          }),
      ),
    );
    const page = create() as unknown as {
      file: File | null;
      upload(): void;
      error(): string | null;
    };
    page.file = new File([new Uint8Array(8)], 'p.pdf', { type: 'application/pdf' });

    page.upload();

    expect(page.error()).toBe('upload exceeds the 100 MB limit');
  });

  it('refuses a file that is not a PDF before uploading anything', () => {
    const page = create() as unknown as {
      onDrop(event: DragEvent): void;
      error(): string | null;
      file: File | null;
    };
    const file = new File([new Uint8Array(4)], 'notes.txt', { type: 'text/plain' });

    page.onDrop({
      preventDefault: () => undefined,
      dataTransfer: { files: [file] },
    } as unknown as DragEvent);

    expect(page.file).toBeNull();
    expect(api.uploadDocument).not.toHaveBeenCalled();
  });

  it('calls a duplicate a duplicate, not a failure', () => {
    const body: UploadResponse = {
      duplicate: true,
      document: { id: 'doc-1', filename: 'prospectus.pdf' } as DocumentRead,
    };
    api.uploadDocument.and.returnValue(of(new HttpResponse({ body })));
    const page = create() as unknown as {
      file: File | null;
      upload(): void;
      notice(): string | null;
      error(): string | null;
    };
    page.file = new File([new Uint8Array(8)], 'prospectus.pdf', { type: 'application/pdf' });

    page.upload();

    expect(page.error()).toBeNull();
    expect(page.notice()).toContain('Already ingested');
  });
});
