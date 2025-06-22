/* docs/js/sidebar.js */
(function(){
  const links = [
    ['Home','index.html'],
    ['API','overview.html'],
    ['Usage','usage.html'],
    ['Quickstart','quickstart.html'],
    ['Install','installation.html'],
    ['Integration','integration.html'],
    ['Optimize','optimization_guide.html'],
    ['Components','component-design.html'],
    ['Cookbook','reward_cookbook.html'],
    ['Concepts','concepts.html'],
    ['With RLlama','with-rllama.html'],
    ['Without RLlama','without-rllama.html'],
    ['Why RLlama','why-rllama.html']
  ];
  // build sidebar
  const nav = document.createElement('nav');
  nav.className = 'sidebar';
  const ul = document.createElement('ul');
  links.forEach(([title,href])=>{
    const a = document.createElement('a');
    a.textContent = title;
    a.href = href;
    if(location.pathname.endsWith(href)) a.classList.add('active');
    const li = document.createElement('li');
    li.appendChild(a);
    ul.appendChild(li);
  });
  nav.appendChild(ul);

  // wrap body content
  const wrapper = document.createElement('div');
  wrapper.className = 'page-with-sidebar';
  const main = document.createElement('main');
  main.className = 'content';
  while(document.body.firstChild){
    main.appendChild(document.body.firstChild);
  }
  wrapper.appendChild(nav);
  wrapper.appendChild(main);
  document.body.appendChild(wrapper);
})();
