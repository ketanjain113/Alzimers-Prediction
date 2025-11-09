/* script.js - unified behavior: theme + active nav highlight + optional small UI helpers */

(function(){
  // THEME: toggle & persist
  const btn = document.querySelectorAll('#theme-toggle');
  const body = document.body;
  const saved = localStorage.getItem('ns_theme') || null;
  if(saved === 'light') body.classList.add('light');

  function updateToggleText(el){
    if(!el) return;
    el.textContent = body.classList.contains('light') ? '☀️' : '🌙';
  }

  // set initial text on all toggles
  document.querySelectorAll('#theme-toggle').forEach(updateToggleText);

  document.addEventListener('click', e=>{
    if(e.target && e.target.id === 'theme-toggle'){
      body.classList.toggle('light');
      const mode = body.classList.contains('light') ? 'light' : 'dark';
      localStorage.setItem('ns_theme', mode);
      document.querySelectorAll('#theme-toggle').forEach(updateToggleText);
    }
  });

  // NAV: active link based on path
  const links = document.querySelectorAll('.nav a');
  const path = location.pathname.split('/').pop() || 'index.html';
  links.forEach(a=>{
    const href = a.getAttribute('href') || '';
    if(href === path || (href === 'index.html' && path === '')) {
      a.classList.add('active');
    } else {
      a.classList.remove('active');
    }
  });

  // small: smooth anchor scrolling if needed
  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    a.addEventListener('click', e=>{ e.preventDefault(); document.querySelector(a.getAttribute('href')).scrollIntoView({behavior:'smooth'}); });
  });

})();
