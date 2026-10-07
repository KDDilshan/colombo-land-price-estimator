/* Site-wide chrome: dark mode toggle, mobile nav, flash message dismissal.
 * Runs on every page (loaded from base.html), independent of the map tool's app.js. */

(function () {
  const root = document.documentElement;

  /* ------------------------------------------------------------- theme */

  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  }

  /* --------------------------------------------------------- mobile nav */

  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      const open = navLinks.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // close the menu after following a link, so it doesn't stay open on the next page
    navLinks.querySelectorAll('a, button[type="submit"]').forEach((el) => {
      el.addEventListener('click', () => {
        navLinks.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ------------------------------------------------------------- flash */

  document.querySelectorAll('.flash-close').forEach((btn) => {
    btn.addEventListener('click', () => {
      const flash = btn.closest('.flash');
      if (flash) flash.remove();
    });
  });
})();
