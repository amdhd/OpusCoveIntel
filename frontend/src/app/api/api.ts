/**
 * The only place this app talks to the server.
 *
 * Every method returns a type from `models.ts`, which comes from the server's
 * own OpenAPI document. Components never build URLs, so "what does this app
 * call?" is answerable by reading one file.
 *
 * No base URL and no `withCredentials`: the app is served from the same origin
 * as the API (`/app` on the FastAPI process, or the dev proxy), which is what
 * lets the session cookie stay `HttpOnly` and `SameSite=lax`. A cross-origin
 * frontend would need `SameSite=none`, and that would give up the CSRF
 * property the server relies on.
 */
import { HttpClient, HttpEvent, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import type {
  CovenantRead,
  DocumentRead,
  DocumentStatusRead,
  InstrumentDetail,
  InstrumentRead,
  LoginResponse,
  PendingResponse,
  QueryResponse,
  ReviewActionResponse,
  UploadResponse,
  UserRead,
} from './models';

@Injectable({ providedIn: 'root' })
export class Api {
  private readonly http = inject(HttpClient);

  // -- auth ------------------------------------------------------------

  login(username: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>('/auth/login', { username, password });
  }

  logout(): Observable<void> {
    return this.http.post<void>('/auth/logout', {});
  }

  me(): Observable<UserRead> {
    return this.http.get<UserRead>('/auth/me');
  }

  // -- documents ---------------------------------------------------------

  listDocuments(limit = 50): Observable<DocumentRead[]> {
    return this.http.get<DocumentRead[]>('/documents', {
      params: new HttpParams().set('limit', limit),
    });
  }

  /**
   * Upload a PDF, reporting progress.
   *
   * `observe: 'events'` rather than a plain response: a 500-page prospectus is
   * tens of megabytes, and a screen that shows nothing until the request
   * finishes is indistinguishable from a screen that has hung.
   */
  uploadDocument(file: File, documentType: string): Observable<HttpEvent<UploadResponse>> {
    const body = new FormData();
    body.append('file', file, file.name);
    body.append('document_type', documentType);
    return this.http.post<UploadResponse>('/documents/upload', body, {
      reportProgress: true,
      observe: 'events',
    });
  }

  documentStatus(documentId: string): Observable<DocumentStatusRead> {
    return this.http.get<DocumentStatusRead>(`/documents/${documentId}/status`);
  }

  /** Parse now rather than waiting for the worker's poll interval. */
  processDocument(documentId: string): Observable<unknown> {
    return this.http.post(`/documents/${documentId}/process`, {});
  }

  // -- catalogue ---------------------------------------------------------

  listInstruments(limit = 100): Observable<InstrumentRead[]> {
    return this.http.get<InstrumentRead[]>('/instruments', {
      params: new HttpParams().set('limit', limit),
    });
  }

  getInstrument(instrumentId: string): Observable<InstrumentDetail> {
    return this.http.get<InstrumentDetail>(`/instruments/${instrumentId}`);
  }

  instrumentCovenants(instrumentId: string): Observable<CovenantRead[]> {
    return this.http.get<CovenantRead[]>(`/instruments/${instrumentId}/covenants`);
  }

  // -- ask ---------------------------------------------------------------

  ask(question: string): Observable<QueryResponse> {
    return this.http.post<QueryResponse>('/query', { question });
  }

  // -- review ------------------------------------------------------------

  pendingReviews(limit = 50): Observable<PendingResponse> {
    return this.http.get<PendingResponse>('/review/pending', {
      params: new HttpParams().set('limit', limit),
    });
  }

  approveReview(reviewId: string, notes?: string): Observable<ReviewActionResponse> {
    return this.http.post<ReviewActionResponse>(`/review/${reviewId}/approve`, {
      notes: notes || null,
    });
  }

  correctReview(
    reviewId: string,
    newValue: string,
    notes?: string,
  ): Observable<ReviewActionResponse> {
    return this.http.post<ReviewActionResponse>(`/review/${reviewId}/correct`, {
      new_value: newValue,
      notes: notes || null,
    });
  }

  rejectReview(reviewId: string, reason?: string): Observable<ReviewActionResponse> {
    return this.http.post<ReviewActionResponse>(`/review/${reviewId}/reject`, {
      reason: reason || null,
      notes: null,
    });
  }
}
