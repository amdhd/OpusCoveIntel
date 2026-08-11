import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { provideRouter, withComponentInputBinding } from '@angular/router';

import { routes } from './app.routes';
import { unauthorizedInterceptor } from './auth/unauthorized-interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideRouter(routes, withComponentInputBinding()),
    // No `withXsrfConfiguration` and no token header: the session cookie is
    // `SameSite=lax`, which is what blocks a cross-site POST from riding it.
    // That only holds while this app is served from the API's own origin --
    // see docs/deploy.md before putting the two on different hosts.
    provideHttpClient(withFetch(), withInterceptors([unauthorizedInterceptor])),
  ],
};
