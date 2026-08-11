/**
 * The chrome around every signed-in screen: brand, nav, who you are.
 *
 * Deliberately the same markup and classes as the server-rendered pages in
 * `app/web/templates/base.html`, and the same stylesheet -- not a copy of it.
 * Two UIs are already one too many; two *looks* would make it obvious.
 */
import { Component, inject, signal } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { Api } from '../api/api';
import { Auth } from '../auth/auth';

@Component({
  selector: 'app-shell',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './shell.html',
})
export class Shell {
  private readonly api = inject(Api);
  private readonly router = inject(Router);
  protected readonly auth = inject(Auth);

  protected readonly pending = signal<number | null>(null);

  constructor() {
    // The queue badge is a count, not a list -- cheap enough to fetch once per
    // load, and the one number a reviewer wants without navigating.
    this.api.pendingReviews(1).subscribe({
      next: (response) => this.pending.set(response.total_pending),
      error: () => this.pending.set(null),
    });
  }

  protected async signOut(): Promise<void> {
    await this.auth.logout();
    await this.router.navigate(['/login']);
  }
}
