/**
 * The session mirror, and the guard that uses it.
 *
 * The property worth pinning is that the server decides: a 401 anywhere means
 * signed out here, and `load()` asks once rather than per navigation.
 */
import { TestBed } from '@angular/core/testing';
import { Router, UrlTree } from '@angular/router';
import { of, throwError } from 'rxjs';

import { HttpErrorResponse } from '@angular/common/http';

import { Api } from '../api/api';
import type { UserRead } from '../api/models';
import { Auth } from './auth';
import { landingGuard } from './auth-guard';

const REVIEWER = { username: 'aminah', display_name: 'Aminah', role: 'reviewer' } as UserRead;

describe('Auth', () => {
  let api: jasmine.SpyObj<Api>;
  let auth: Auth;

  beforeEach(() => {
    api = jasmine.createSpyObj<Api>('Api', ['me', 'login', 'logout']);
    TestBed.configureTestingModule({ providers: [{ provide: Api, useValue: api }] });
    auth = TestBed.inject(Auth);
  });

  it('asks the server once per page load, not once per navigation', async () => {
    api.me.and.returnValue(of(REVIEWER));

    await auth.load();
    await auth.load();
    await auth.load();

    expect(api.me).toHaveBeenCalledTimes(1);
    expect(auth.signedIn()).toBeTrue();
  });

  it('treats a 401 from /auth/me as "not signed in", not as an error', async () => {
    api.me.and.returnValue(throwError(() => new HttpErrorResponse({ status: 401 })));

    const user = await auth.load();

    expect(user).toBeNull();
    expect(auth.signedIn()).toBeFalse();
  });

  it('only an analyst who is a reviewer may be offered review controls', async () => {
    api.me.and.returnValue(of({ ...REVIEWER, role: 'analyst' } as UserRead));

    await auth.load();

    expect(auth.signedIn()).toBeTrue();
    expect(auth.canReview()).toBeFalse();
  });

  it('lands a reviewer on the queue and everyone else on the corpus', async () => {
    const router = TestBed.inject(Router);
    const landing = async (role: string): Promise<string> => {
      api.me.calls.reset();
      api.me.and.returnValue(of({ ...REVIEWER, role } as UserRead));
      TestBed.resetTestingModule();
      TestBed.configureTestingModule({ providers: [{ provide: Api, useValue: api }] });
      await TestBed.inject(Auth).load();
      return router.serializeUrl(
        TestBed.runInInjectionContext(() => landingGuard({} as never, {} as never)) as UrlTree,
      );
    };

    // A reviewer is here to clear the queue, including on the day it is empty:
    // "nothing pending" is the answer they came for.
    expect(await landing('reviewer')).toBe('/review');
    expect(await landing('analyst')).toBe('/documents');
  });

  it('forgets the user even when the logout request fails', async () => {
    api.me.and.returnValue(of(REVIEWER));
    api.logout.and.returnValue(throwError(() => new HttpErrorResponse({ status: 500 })));
    await auth.load();

    await auth.logout();

    // A stale name in the corner of the screen is worse than an extra login.
    expect(auth.signedIn()).toBeFalse();
  });
});
