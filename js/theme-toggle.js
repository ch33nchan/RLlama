/* docs/js/theme-toggle.js */
(function(){
  const wrapper = document.createElement('div');
  wrapper.className = 'theme-switch';
  wrapper.innerHTML = '<input type="checkbox" id="theme-toggle"><label for="theme-toggle">Dark Mode</label>';
  document.body.appendChild(wrapper);

  const input = wrapper.querySelector('input');
  const setTheme = theme => {
    document.documentElement.setAttribute('data-theme', theme);
    input.checked = theme === 'dark';
    localStorage.setItem('theme', theme);
  };

  input.addEventListener('change', () => {
    setTheme(input.checked ? 'dark' : 'light');
  });

  const saved = localStorage.getItem('theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  setTheme(saved);
})();
