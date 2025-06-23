/* docs/js/home-enhance.js */
(function(){
  if (!location.pathname.endsWith('index.html')) return;
  const main = document.querySelector('main.content');
  if (!main) return;

  const about = document.createElement('section');
  about.style.margin = '2rem 0';
  about.innerHTML = `
    <h2>About RLlama</h2>
    <p>RLlama is a Python library that makes reward engineering simple, composable, and optimizable. Combine multiple reward components, search for the best weights, and visualize contributions—all in one toolkit.</p>
  `;
  main.insertBefore(about, main.querySelector('.features'));
})();
