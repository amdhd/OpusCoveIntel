/**
 * Routes.
 *
 * Every screen except the login form sits behind `authGuard` and inside the
 * shell, so a route added here is authenticated and framed by default -- the
 * same reasoning as the server's document router, which puts
 * `Depends(current_user)` on the router rather than on each handler.
 *
 * Lazily loaded: a browser that only ever visits the upload screen should not
 * download the review queue.
 */
import { Routes } from '@angular/router';

import { authGuard } from './auth/auth-guard';

export const routes: Routes = [
  {
    path: 'login',
    loadComponent: () => import('./auth/login.page').then((m) => m.LoginPage),
    title: 'Sign in — OpusCovIntel',
  },
  {
    path: '',
    loadComponent: () => import('./shell/shell').then((m) => m.Shell),
    canActivate: [authGuard],
    children: [
      {
        path: 'documents',
        loadComponent: () => import('./documents/documents.page').then((m) => m.DocumentsPage),
        title: 'Documents — OpusCovIntel',
      },
      {
        path: 'ask',
        loadComponent: () => import('./ask/ask.page').then((m) => m.AskPage),
        title: 'Ask — OpusCovIntel',
      },
      {
        path: 'instruments',
        loadComponent: () =>
          import('./instruments/instruments.page').then((m) => m.InstrumentsPage),
        title: 'Instruments — OpusCovIntel',
      },
      {
        path: 'review',
        loadComponent: () => import('./review/review.page').then((m) => m.ReviewPage),
        title: 'Review — OpusCovIntel',
      },
      { path: '', pathMatch: 'full', redirectTo: 'documents' },
    ],
  },
  { path: '**', redirectTo: '' },
];
