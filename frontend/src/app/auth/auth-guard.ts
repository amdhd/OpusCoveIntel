/**
 * Route guard: no screen renders for a visitor the server does not know.
 *
 * This is convenience, not security. Every endpoint behind these screens is
 * authenticated on the server, and a guard in the browser is a thing anyone
 * can switch off in a debugger. What it buys is that an expired session shows
 * a login form instead of four panels of failed requests.
 */
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';

import { Auth } from './auth';

export const authGuard: CanActivateFn = async (_route, state) => {
  const auth = inject(Auth);
  const router = inject(Router);

  if (await auth.load()) {
    return true;
  }
  // Carry where they were going, so signing in lands them there rather than on
  // a default screen they did not ask for.
  return router.createUrlTree(['/login'], { queryParams: { next: state.url } });
};

/**
 * Where `/app` with no path lands you: the screen your role is there to work.
 *
 * A reviewer exists to clear the queue, so the queue is their front page --
 * including when it is empty, which is the answer they wanted. Everyone else
 * lands on the corpus.
 *
 * A guard rather than a functional `redirectTo`, and deliberately: `redirectTo`
 * is evaluated during route recognition, which happens *before* `authGuard`
 * has awaited `/auth/me`, so it would read a role that is still null on a cold
 * load. Child guards run after the parent's, by which point the role is known.
 */
export const landingGuard: CanActivateFn = () => {
  const auth = inject(Auth);
  const router = inject(Router);

  return router.createUrlTree([auth.canReview() ? '/review' : '/documents']);
};
