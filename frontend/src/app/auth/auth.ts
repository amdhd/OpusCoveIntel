/**
 * Who is signed in, as far as this browser tab knows.
 *
 * The session itself lives in an `HttpOnly` cookie the server set, so this app
 * cannot read it and does not try. `user()` is a cache of the answer to
 * `GET /auth/me`, refreshed on load and cleared whenever the server says 401 --
 * the server is the authority on the session, and the client only mirrors it.
 * Storing a token in `localStorage` would be readable by any script on the
 * page, which is precisely what `HttpOnly` exists to prevent.
 */
import { Injectable, computed, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

import { Api } from '../api/api';
import type { UserRead } from '../api/models';

@Injectable({ providedIn: 'root' })
export class Auth {
  private readonly api = inject(Api);

  private readonly current = signal<UserRead | null>(null);
  private resolved = false;

  readonly user = this.current.asReadonly();
  readonly signedIn = computed(() => this.current() !== null);
  readonly canReview = computed(() => this.current()?.role === 'reviewer');

  /**
   * Ask the server who we are, once per page load.
   *
   * Returns the cached answer afterwards so a guard on every route does not
   * make a request per navigation.
   */
  async load(): Promise<UserRead | null> {
    if (this.resolved) {
      return this.current();
    }
    try {
      this.current.set(await firstValueFrom(this.api.me()));
    } catch {
      // 401 is the normal answer for a visitor who is not signed in, so it is
      // not logged as an error.
      this.current.set(null);
    }
    this.resolved = true;
    return this.current();
  }

  async login(username: string, password: string): Promise<UserRead> {
    const response = await firstValueFrom(this.api.login(username, password));
    this.current.set(response.user);
    this.resolved = true;
    return response.user;
  }

  async logout(): Promise<void> {
    try {
      await firstValueFrom(this.api.logout());
    } catch {
      // Swallowed, not rethrown. The caller's next act is to navigate to the
      // login screen, and a rejected promise here would skip it -- leaving
      // someone who asked to sign out looking at a signed-in page. The cookie
      // may well be gone anyway; the server is idempotent about logout.
    }
    // Cleared either way: the intent was to stop being signed in here, and a
    // stale name in the corner of the screen is worse than an extra login.
    this.forget();
  }

  /** Drop the cached identity. Called on logout and on any 401. */
  forget(): void {
    this.current.set(null);
    this.resolved = true;
  }
}
