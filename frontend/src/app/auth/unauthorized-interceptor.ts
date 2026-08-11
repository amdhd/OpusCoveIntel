/**
 * A 401 from anywhere means the session ended. Say so once, in one place.
 *
 * Sessions are revocable rows: an operator deactivating an account, or a
 * password change, invalidates a live session mid-visit. Without this, the
 * screen keeps its stale name in the corner and every panel quietly fails.
 *
 * `/auth/me` is exempt because a 401 there is the expected answer for a
 * visitor who is not signed in -- redirecting on it would bounce the login
 * page to itself.
 */
import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { Auth } from './auth';

export const unauthorizedInterceptor: HttpInterceptorFn = (request, next) => {
  const auth = inject(Auth);
  const router = inject(Router);

  return next(request).pipe(
    catchError((error: unknown) => {
      const isSessionProbe = request.url.endsWith('/auth/me');
      if (error instanceof HttpErrorResponse && error.status === 401 && !isSessionProbe) {
        auth.forget();
        void router.navigate(['/login'], { queryParams: { next: router.url } });
      }
      return throwError(() => error);
    }),
  );
};
