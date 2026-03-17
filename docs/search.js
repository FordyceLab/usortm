/* uSort-M documentation site search
 * Builds its index by fetching all pages at runtime — no static index to maintain.
 * Add new pages to the PAGES array when they are created.
 */
(function () {

  // ── Pages to index ──────────────────────────────────────────────────────────
  // Add an entry here whenever a new doc page is created.
  var PAGES = [
    { name: 'Overview',        url: 'index.html' },
    { name: 'Getting Started', url: 'getting-started.html' },
    { name: 'Workflow',        url: 'workflow.html' },
    { name: 'Library Design',  url: 'library-design.html' },
    { name: 'FACS Sorting',    url: 'sorting.html' },
    { name: 'PCR Barcoding',   url: 'barcoding.html' },
    { name: 'Demultiplexing',  url: 'demultiplexing.html' },
    { name: 'CLI Reference',   url: 'cli.html' },
    { name: 'Python API',      url: 'api.html' },
  ];

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/^\d+\.\s*/, '')       // strip leading "1. " from protocol steps
      .replace(/[^a-z0-9\s]/g, ' ')   // non-alphanumeric → space
      .trim()
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');
  }

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── Index building ──────────────────────────────────────────────────────────
  var index = null;   // null = not yet built
  var building = null; // Promise while build is in progress

  function parsePageForIndex(html, pageName, pageUrl) {
    var parser = new DOMParser();
    var doc = parser.parseFromString(html, 'text/html');
    var entries = [];

    // Page-level entry
    entries.push({ title: pageName, page: pageName, url: pageUrl, snippet: '' });

    // Section entries from h2 and h3 inside .main
    var headings = doc.querySelectorAll('.main h2, .main h3');
    headings.forEach(function (h) {
      var title = h.textContent.trim();
      if (!title) return;
      var id = h.id || slugify(title);

      // Grab a short snippet from the next sibling <p>
      var snippet = '';
      var next = h.nextElementSibling;
      while (next && !snippet) {
        if (next.tagName === 'P') {
          snippet = next.textContent.trim().slice(0, 120);
        }
        next = next.nextElementSibling;
      }

      entries.push({ title: title, page: pageName, url: pageUrl + '#' + id, snippet: snippet });
    });

    return entries;
  }

  function buildIndex() {
    if (index !== null) return Promise.resolve();
    if (building) return building;

    // Try sessionStorage cache first
    try {
      var cached = sessionStorage.getItem('usortm-search-v1');
      if (cached) {
        index = JSON.parse(cached);
        return Promise.resolve();
      }
    } catch (e) {}

    var baseUrl = window.location.href.replace(/[^/]*(\?.*)?$/, '');

    building = Promise.all(
      PAGES.map(function (page) {
        return fetch(baseUrl + page.url)
          .then(function (res) { return res.ok ? res.text() : ''; })
          .then(function (html) {
            return html ? parsePageForIndex(html, page.name, page.url) : [];
          })
          .catch(function () { return []; });
      })
    ).then(function (results) {
      index = [].concat.apply([], results);
      try { sessionStorage.setItem('usortm-search-v1', JSON.stringify(index)); } catch (e) {}
      building = null;
    });

    return building;
  }

  // ── Search ──────────────────────────────────────────────────────────────────
  function search(query) {
    if (!index || !query.trim()) return [];
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);

    return index
      .map(function (entry) {
        var titleLow   = entry.title.toLowerCase();
        var pageLow    = entry.page.toLowerCase();
        var snippetLow = (entry.snippet || '').toLowerCase();

        // All terms must appear somewhere
        var allFound = terms.every(function (t) {
          return titleLow.includes(t) || pageLow.includes(t) || snippetLow.includes(t);
        });
        if (!allFound) return null;

        // Score: higher for title hits
        var score = terms.reduce(function (s, t) {
          if (titleLow.includes(t))   return s + 4;
          if (pageLow.includes(t))    return s + 2;
          if (snippetLow.includes(t)) return s + 1;
          return s;
        }, 0);

        return { entry: entry, score: score };
      })
      .filter(Boolean)
      .sort(function (a, b) { return b.score - a.score; })
      .slice(0, 8)
      .map(function (r) { return r.entry; });
  }

  // ── Heading ID injection (current page) ────────────────────────────────────
  function injectHeadingIds() {
    document.querySelectorAll('.main h2, .main h3').forEach(function (h) {
      if (!h.id) h.id = slugify(h.textContent.trim());
    });
  }

  // ── UI ──────────────────────────────────────────────────────────────────────
  function initSearch() {
    var wrapper = document.getElementById('site-search-wrapper');
    if (!wrapper) return;

    var input   = document.getElementById('site-search-input');
    var results = document.getElementById('site-search-results');
    var activeIdx = -1;

    function setActive(idx) {
      var items = results.querySelectorAll('.search-result-item');
      items.forEach(function (el, i) { el.classList.toggle('active', i === idx); });
      activeIdx = idx;
    }

    function hideResults() {
      results.style.display = 'none';
      results.innerHTML = '';
      activeIdx = -1;
    }

    function renderLoading() {
      results.innerHTML = '<div class="search-loading"><span class="search-spinner"></span>Indexing…</div>';
      results.style.display = 'block';
    }

    function renderResults(items) {
      results.innerHTML = '';
      activeIdx = -1;
      if (!items.length) {
        results.innerHTML = '<div class="search-empty">No results</div>';
        results.style.display = 'block';
        return;
      }
      items.forEach(function (item) {
        var a = document.createElement('a');
        a.href = item.url;
        a.className = 'search-result-item';
        a.innerHTML =
          '<span class="search-result-page">' + escHtml(item.page) + '</span>' +
          '<span class="search-result-title">' + escHtml(item.title) + '</span>';
        a.addEventListener('mousedown', function (e) { e.preventDefault(); });
        a.addEventListener('click', function () { hideResults(); input.value = ''; });
        results.appendChild(a);
      });
      results.style.display = 'block';
    }

    function doSearch(q) {
      if (!q.trim()) { hideResults(); return; }
      if (index === null) {
        renderLoading();
        buildIndex().then(function () { renderResults(search(q)); });
      } else {
        renderResults(search(q));
      }
    }

    input.addEventListener('input', function () { doSearch(input.value); });

    input.addEventListener('keydown', function (e) {
      var items = results.querySelectorAll('.search-result-item');
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActive(Math.min(activeIdx + 1, items.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActive(Math.max(activeIdx - 1, 0));
      } else if (e.key === 'Enter' && activeIdx >= 0 && items[activeIdx]) {
        window.location.href = items[activeIdx].href;
        hideResults();
        input.value = '';
      } else if (e.key === 'Escape') {
        hideResults();
        input.blur();
      }
    });

    input.addEventListener('focus', function () {
      buildIndex(); // start prefetching index on first focus (non-blocking)
      if (input.value.trim()) doSearch(input.value);
    });

    input.addEventListener('blur', function () {
      setTimeout(hideResults, 150);
    });

    document.addEventListener('click', function (e) {
      if (!wrapper.contains(e.target)) hideResults();
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    injectHeadingIds();
    initSearch();
  });

})();
