/* docs/js/sidebar.js */
(function(){
  // build the sidebar menu
  var links = [
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
  var nav = document.createElement('nav');
  nav.className = 'sidebar';
  var ul = document.createElement('ul');
  links.forEach(function(item){
    var a = document.createElement('a');
    a.textContent = item[0];
    a.href = item[1];
    if(location.pathname.endsWith(item[1])) a.classList.add('active');
    var li = document.createElement('li');
    li.append(a);
    ul.append(li);
  });
  nav.append(ul);

  // wrap existing body in a flex container
  var wrapper = document.createElement('div');
  wrapper.className = 'page-with-sidebar';
  // move all children into <main>
  var main = document.createElement('main');
  Array.from(document.body.childNodes).forEach(function(node){
    main.append(node);
  });
  wrapper.append(nav, main);
  document.body.append(wrapper);
})();
