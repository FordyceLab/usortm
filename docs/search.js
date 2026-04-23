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
    { name: 'Hit Picking',     url: 'hitpicking.html' },
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

  function escRegExp(s) {
    return s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  }

  // Wrap every case-insensitive occurrence of any term in <mark>.
  // Input must already be HTML-escaped (tags in terms would otherwise survive).
  function highlightTerms(escapedHtml, terms) {
    if (!terms.length) return escapedHtml;
    var pattern = terms.map(escRegExp).join('|');
    return escapedHtml.replace(new RegExp('(' + pattern + ')', 'gi'), '<mark>$1</mark>');
  }

  // Extract a ~120-char window around the first term match, with ellipses
  // for truncation and each term occurrence wrapped in <mark>. Returns ''
  // if the text contains no term hits (title-only matches).
  function buildSnippet(text, terms, windowLen) {
    windowLen = windowLen || 120;
    if (!text || !terms.length) return '';
    var textLow = text.toLowerCase();
    var firstIdx = -1;
    terms.forEach(function (t) {
      var i = textLow.indexOf(t);
      if (i >= 0 && (firstIdx === -1 || i < firstIdx)) firstIdx = i;
    });
    if (firstIdx === -1) return '';

    // Favor context AFTER the match over context before it.
    var prefixLen = Math.floor(windowLen / 3);
    var start = Math.max(0, firstIdx - prefixLen);
    var end = Math.min(text.length, start + windowLen);

    // Snap to nearest word boundary so we don't start/end mid-word.
    if (start > 0) {
      var sp = text.indexOf(' ', start);
      if (sp > 0 && sp - start < 15) start = sp + 1;
    }
    if (end < text.length) {
      var sp2 = text.lastIndexOf(' ', end);
      if (sp2 > start + windowLen / 2) end = sp2;
    }

    var prefix = start > 0 ? '…' : '';
    var suffix = end < text.length ? '…' : '';
    return prefix + highlightTerms(escHtml(text.slice(start, end)), terms) + suffix;
  }

  // ── Index building ──────────────────────────────────────────────────────────
  var index = null;   // null = not yet built
  var building = null; // Promise while build is in progress

  function parsePageForIndex(html, pageName, pageUrl) {
    var parser = new DOMParser();
    var doc = parser.parseFromString(html, 'text/html');
    var entries = [{ title: pageName, page: pageName, url: pageUrl, text: '' }];

    var main = doc.querySelector('.main');
    if (!main) return entries;

    // Walk all content nodes in document order, grouping text under headings.
    // Collect p, li, td, dt, dd — block-level text elements that won't double-count
    // text already gathered from a parent element.
    var nodes = main.querySelectorAll('h2, h3, p, li, td, dt, dd');
    var sections = [];
    var cur = null;

    nodes.forEach(function (node) {
      var tag = node.tagName;
      if (tag === 'H2' || tag === 'H3') {
        if (cur) sections.push(cur);
        var title = node.textContent.trim();
        cur = { title: title, id: node.id || slugify(title), parts: [] };
      } else if (cur) {
        var text = node.textContent.trim();
        if (text) cur.parts.push(text);
      }
    });
    if (cur) sections.push(cur);

    sections.forEach(function (s) {
      entries.push({
        title: s.title,
        page: pageName,
        url: pageUrl + '#' + s.id,
        text: s.parts.join(' ').replace(/\s+/g, ' ').slice(0, 4000),
      });
    });

    return entries;
  }

  function buildIndex() {
    if (index !== null) return Promise.resolve();
    if (building) return building;

    // Try sessionStorage cache first
    try {
      var cached = sessionStorage.getItem('usortm-search-v2');
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
      try { sessionStorage.setItem('usortm-search-v2', JSON.stringify(index)); } catch (e) {}
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
        var titleLow = entry.title.toLowerCase();
        var pageLow  = entry.page.toLowerCase();
        var textLow  = (entry.text || '').toLowerCase();

        // All terms must appear somewhere
        var allFound = terms.every(function (t) {
          return titleLow.includes(t) || pageLow.includes(t) || textLow.includes(t);
        });
        if (!allFound) return null;

        // Score: title hits rank highest, then page name, then body text
        var score = terms.reduce(function (s, t) {
          if (titleLow.includes(t)) return s + 4;
          if (pageLow.includes(t))  return s + 2;
          if (textLow.includes(t))  return s + 1;
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

    function renderResults(items, terms) {
      results.innerHTML = '';
      activeIdx = -1;
      if (!items.length) {
        results.innerHTML = '<div class="search-empty">No results</div>';
        results.style.display = 'block';
        return;
      }
      terms = terms || [];
      items.forEach(function (item) {
        var a = document.createElement('a');
        a.href = item.url;
        a.className = 'search-result-item';
        var titleHtml = highlightTerms(escHtml(item.title), terms);
        var snippetHtml = buildSnippet(item.text || '', terms);
        a.innerHTML =
          '<span class="search-result-page">' + escHtml(item.page) + '</span>' +
          '<span class="search-result-title">' + titleHtml + '</span>' +
          (snippetHtml ? '<span class="search-result-snippet">' + snippetHtml + '</span>' : '');
        a.addEventListener('mousedown', function (e) { e.preventDefault(); });
        a.addEventListener('click', function () { hideResults(); input.value = ''; });
        results.appendChild(a);
      });
      results.style.display = 'block';
    }

    function doSearch(q) {
      if (!q.trim()) { hideResults(); return; }
      var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
      if (index === null) {
        renderLoading();
        buildIndex().then(function () { renderResults(search(q), terms); });
      } else {
        renderResults(search(q), terms);
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

  // ── Mobile nav toggle ───────────────────────────────────────────────────────
  function initNavToggle() {
    var sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    var btn = document.createElement('button');
    btn.className = 'nav-toggle';
    btn.setAttribute('aria-label', 'Toggle navigation');
    btn.innerHTML =
      '<svg class="icon-open" width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">' +
        '<path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>' +
      '</svg>' +
      '<svg class="icon-close" width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">' +
        '<path d="M4 4l12 12M16 4L4 16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>' +
      '</svg>';

    var header = sidebar.querySelector('.sidebar-header');
    if (header) {
      header.insertAdjacentElement('afterend', btn);
    } else {
      sidebar.insertBefore(btn, sidebar.firstChild);
    }

    function close() { sidebar.classList.remove('nav-open'); }

    btn.addEventListener('click', function () {
      sidebar.classList.toggle('nav-open');
    });

    // Close when a nav link is clicked
    sidebar.querySelectorAll('.sidebar-nav a').forEach(function (a) {
      a.addEventListener('click', close);
    });

    // Close on Escape
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') close();
    });

    // Close on click outside sidebar
    document.addEventListener('click', function (e) {
      if (!sidebar.contains(e.target)) close();
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    injectHeadingIds();
    initSearch();
    initNavToggle();
  });

})();
