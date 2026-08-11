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
