/**
 * Sign in.
 *
 * One message for every failure, matching the server: a wrong username, a
 * wrong password and a deactivated account are indistinguishable here because
 * they are indistinguishable there, and a client that distinguished them would
 * hand back the username oracle the server works to deny.
 *
 * 429 is the exception, and it has to be: a rate limit that cannot be observed
 * cannot be obeyed. It says how long to wait and nothing about who exists.
 */
import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { Auth } from './auth';

@Component({
  selector: 'app-login',
  imports: [FormsModule],
  templateUrl: './login.page.html',
})
export class LoginPage {
  private readonly auth = inject(Auth);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  protected username = '';
  protected password = '';
  protected readonly error = signal<string | null>(null);
  protected readonly busy = signal(false);

  protected async submit(): Promise<void> {
    this.busy.set(true);
    this.error.set(null);
    try {
      await this.auth.login(this.username, this.password);
      await this.router.navigateByUrl(this.destination());
    } catch (caught) {
      this.error.set(this.describe(caught));
    } finally {
      this.busy.set(false);
    }
  }

  /**
   * Where to go after signing in.
   *
   * Only a path within this app, never a full URL from the query string: the
   * server's own login does the same, because a `next` a caller controls is an
   * open redirect.
   */
  private destination(): string {
    const next = this.route.snapshot.queryParamMap.get('next') ?? '';
    return next.startsWith('/') && !next.startsWith('//') ? next : '/documents';
  }

  private describe(caught: unknown): string {
    if (caught instanceof HttpErrorResponse && caught.status === 429) {
      const retryAfter = caught.headers.get('Retry-After');
      return retryAfter
        ? `Too many failed sign-in attempts. Try again in ${retryAfter} seconds.`
        : 'Too many failed sign-in attempts. Try again shortly.';
    }
    if (caught instanceof HttpErrorResponse && caught.status === 0) {
      return 'Cannot reach the server.';
    }
    return 'Invalid username or password.';
  }
}
