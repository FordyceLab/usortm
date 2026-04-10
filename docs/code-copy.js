// Scroll-fade indicators for wide tables
document.addEventListener('DOMContentLoaded', function() {
  function updateScrollFade(wrapper, container) {
    var tol = 2;
    var canLeft = wrapper.scrollLeft > tol;
    var canRight = wrapper.scrollLeft + wrapper.clientWidth < wrapper.scrollWidth - tol;
    container.classList.toggle('can-scroll-left', canLeft);
    container.classList.toggle('can-scroll-right', canRight);
  }

  document.querySelectorAll('.table-wrapper').forEach(function(w) {
    // Wrap in a container so pseudo-elements overlay correctly
    var container = document.createElement('div');
    container.className = 'table-scroll-container';
    w.parentNode.insertBefore(container, w);
    container.appendChild(w);

    updateScrollFade(w, container);
    w.addEventListener('scroll', function() { updateScrollFade(w, container); }, { passive: true });
  });

  // Re-check after fonts/images settle
  window.addEventListener('load', function() {
    document.querySelectorAll('.table-scroll-container').forEach(function(c) {
      var w = c.querySelector('.table-wrapper');
      if (w) updateScrollFade(w, c);
    });
  });
});

// Add copy buttons to all code blocks
document.addEventListener('DOMContentLoaded', function() {
  // Find all pre elements with code
  const codeBlocks = document.querySelectorAll('pre');

  codeBlocks.forEach(function(pre) {
    // Create copy button
    const button = document.createElement('button');
    button.className = 'code-copy-btn';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code to clipboard');

    // Add click handler
    button.addEventListener('click', async function() {
      const code = pre.querySelector('code');
      const text = code ? code.textContent : pre.textContent;

      try {
        await navigator.clipboard.writeText(text);
        button.textContent = 'Copied!';
        button.classList.add('copied');

        // Reset after 2 seconds
        setTimeout(function() {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
        button.textContent = 'Failed';
        setTimeout(function() {
          button.textContent = 'Copy';
        }, 2000);
      }
    });

    // Add button to pre element
    pre.appendChild(button);
  });
});
