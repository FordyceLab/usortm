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
