import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

/**
 * Root component. Deliberately empty: the chrome lives in `Shell`, which is a
 * routed component, so the login screen renders without a nav bar offering
 * links a signed-out visitor cannot follow.
 */
@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  template: '<router-outlet />',
})
export class App {}
