// static/js/mermaid-init.js
document.addEventListener('DOMContentLoaded', () => {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {
      htmlLabels: true,
      curve: 'linear',
    },
  });
  
  // Find all Mermaid code blocks and render them
  document.querySelectorAll('.mermaid').forEach((element) => {
    mermaid.init(undefined, element);
  });
});