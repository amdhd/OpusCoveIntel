// Karma, with a headless launcher that also works in a container.
//
// The Angular CLI's default `ChromeHeadless` assumes a sandbox the GitHub
// runner's Chrome cannot always create, and the failure is a timeout with no
// explanation. `--no-sandbox` is safe here and only here: the browser runs our
// own test bundle, on a throwaway machine, with no user data.
//
// `frameworks` and `plugins` are listed explicitly: supplying a config file
// replaces the builder's defaults rather than extending them, and without
// these the bundle loads into a page where `describe` does not exist.
module.exports = function (config) {
  config.set({
    frameworks: ['jasmine'],
    plugins: [
      require('karma-jasmine'),
      require('karma-chrome-launcher'),
      require('karma-jasmine-html-reporter'),
    ],
    browsers: ['ChromeHeadlessCI'],
    customLaunchers: {
      ChromeHeadlessCI: {
        base: 'ChromeHeadless',
        flags: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
      },
    },
    reporters: ['progress'],
    restartOnFileChange: true,
  });
};
