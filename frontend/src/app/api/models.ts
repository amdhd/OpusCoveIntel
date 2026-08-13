/**
 * Wire types, taken from the server rather than retyped.
 *
 * `schema.d.ts` is generated from the API's own OpenAPI document, which is
 * generated from the Pydantic models -- so a field renamed in Python breaks
 * the TypeScript build rather than becoming `undefined` at runtime in front of
 * a credit analyst. Regenerate with `make frontend-types`; CI fails if the
 * committed copy has drifted.
 *
 * This file only gives the generated names something readable to import.
 */
import type { components } from './schema';

type Schemas = components['schemas'];

export type UserRead = Schemas['UserRead'];
export type LoginResponse = Schemas['LoginResponse'];

export type DocumentRead = Schemas['DocumentRead'];
export type DocumentStatusRead = Schemas['DocumentStatusRead'];
export type IngestionJobRead = Schemas['IngestionJobRead'];
export type UploadResponse = Schemas['UploadResponse'];
export type DocumentStatus = Schemas['DocumentStatus'];
export type DocumentType = Schemas['DocumentType'];

export type InstrumentRead = Schemas['InstrumentRead'];
export type InstrumentDetail = Schemas['InstrumentDetail'];
export type CovenantRead = Schemas['CovenantRead'];
export type PortfolioRead = Schemas['PortfolioRead'];

export type QueryResponse = Schemas['QueryResponse'];
export type Citation = Schemas['Citation'];

export type PendingResponse = Schemas['PendingResponse'];
export type ReviewItem = Schemas['ReviewItem'];
export type ReviewActionResponse = Schemas['ReviewActionResponse'];

/**
 * Statuses the server calls terminal.
 *
 * Mirrored here only for labelling; the poller stops on the server's
 * `terminal` flag, never on this list. Two places deciding when ingestion is
 * finished is exactly how a client ends up reporting a half-parsed document as
 * done.
 */
export const FAILED_STATUSES: ReadonlySet<string> = new Set(['failed', 'budget_exceeded']);

/**
 * Statuses in which a question can actually reach a document.
 *
 * The status endpoint computes this server-side and the detail panel uses that
 * answer. This copy exists for the corpus list, whose rows are `DocumentRead`
 * and carry only a status -- and a list that shows an ingested-but-unindexed
 * document identically to an indexed one is how three real prospectuses sat
 * invisible while every question about them was answered from elsewhere.
 */
export const SEARCHABLE_STATUSES: ReadonlySet<string> = new Set([
  'embedded',
  'extracting',
  'extracted',
]);
