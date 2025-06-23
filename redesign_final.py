import os
import shutil
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def apply_final_design():
    # --- Configuration ---
    backup_dir = 'docs_backup'
    new_site_dir = 'site'
    
    # --- Clean up old 'site' directory ---
    if os.path.exists(new_site_dir):
        shutil.rmtree(new_site_dir)
    os.makedirs(os.path.join(new_site_dir, 'css'))
    os.makedirs(os.path.join(new_site_dir, 'js'))

    all_pages = sorted([f for f in os.listdir(backup_dir) if f.endswith('.html')])
    pygments_css = HtmlFormatter(style='one-dark').get_style_defs('.code-block')

    for filename in all_pages:
        if not os.path.exists(os.path.join(backup_dir, filename)):
            continue
            
        with open(os.path.join(backup_dir, filename), 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        old_content = soup.find('main', class_='content') or soup.body
        if old_content:
            for tag in old_content.find_all(['nav', 'div'], class_=['sidebar', 'theme-switch']):
                tag.decompose()

        is_homepage = (filename == 'index.html')
        page_layout_class = "layout-homepage" if is_homepage else "layout-docs"

        # --- New HTML Structure ---
        new_soup = BeautifulSoup(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLlama - {filename.replace('.html', '').capitalize()}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="theme-switcher">
        <button class="theme-btn" data-theme="light" title="Light Theme"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg></button>
        <button class="theme-btn" data-theme="dark" title="Dark Theme"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg></button>
    </div>
    <div class="page-container {page_layout_class}">
        <aside class="sidebar">
            <nav id="new-sidebar-nav"></nav>
        </aside>
        <main class="main-content">
            <header id="page-header"></header>
            <div id="content-container"></div>
        </main>
        {"<aside class='right-pane'><canvas id='particle-canvas'></canvas></aside>" if is_homepage else ""}
    </div>
    <footer class="main-footer">
        <p>engineered and designed by <a href="https://srinivastb.netlify.app/" target="_blank">srini</a> with <3</p>
    </footer>
    <script src="js/main.js"></script>
</body>
</html>
        """, 'html.parser')

        # --- Populate Sidebar ---
        sidebar_nav = new_soup.find(id='new-sidebar-nav')
        sidebar_nav.append(BeautifulSoup('<a href="index.html" class="nav-link nav-title">RLlama</a>', 'html.parser'))
        sidebar_nav.append(BeautifulSoup('<hr/>', 'html.parser'))
        for page in all_pages:
            if page != 'index.html':
                page_name = page.replace('.html', '').replace('_', ' ').capitalize()
                sidebar_nav.append(BeautifulSoup(f'<a href="{page}" class="nav-link">{page_name}</a>', 'html.parser'))

        # --- Inject Content and Header ---
        content_container = new_soup.find(id='content-container')
        content_container.append(old_content)
        header = new_soup.find(id='page-header')

        if is_homepage:
            header.append(BeautifulSoup('<h1><span class="scratch" data-text="Struggling with rewards?">RLlama is here to help.</span></h1>', 'html.parser'))
            
            intro_p = "<p class='intro-text'>A Python library for designing, composing, and tuning reward functions in RL. Mix components, auto-search weights, and visualize term contributions.</p>"
            buttons = """
            <div class='homepage-links'>
                <a href='https://pypi.org/project/rllama/' target='_blank' class='homepage-btn'>PyPI</a>
                <a href='https://github.com/ch33nchan/RLlama/' target='_blank' class='homepage-btn'>GitHub</a>
            </div>
            """
            header.insert_after(BeautifulSoup(intro_p + buttons, 'html.parser'))

            if old_content:
                features_section = old_content.find('section', class_='features')
                if features_section:
                    features_section['class'] = features_section.get('class', []) + ['feature-boxes']
                    if features_section.find('h2'):
                        features_section.find('h2').decompose()
        else:
            page_title = filename.replace('.html', '').replace('_', ' ').capitalize()
            if old_content and old_content.h1:
                old_content.h1.decompose()
            header.append(BeautifulSoup(f"<h1>{page_title}</h1>", 'html.parser'))

        # --- Process Code Blocks with Pygments ---
        for pre_tag in new_soup.find_all('pre'):
            code_tag = pre_tag.find('code')
            if code_tag:
                code_content = code_tag.get_text()
                highlighted_code = highlight(code_content, PythonLexer(), HtmlFormatter(style='one-dark', nowrap=True))
                
                new_code_container = BeautifulSoup(f'<div class="code-block">{highlighted_code}</div>', 'html.parser')
                
                highlight_div = new_code_container.find('div', class_='highlight')
                if highlight_div:
                    highlight_div.wrap(new_soup.new_tag("div", **{'class': 'code-scroll-container'}))
                
                new_code_container.append(BeautifulSoup('<button class="copy-button" title="Copy code"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></button>', 'html.parser'))
                
                pre_tag.replace_with(new_code_container)
        
        with open(os.path.join(new_site_dir, filename), 'w') as f:
            f.write(str(new_soup.prettify()))
    
    # --- Create NEW CSS file ---
    css_content = pygments_css + """
    :root { --font-main: 'Inter', sans-serif; }
    :root[data-theme="light"] {
        --bg: #fff; --fg: #000; --border: #eaeaea; --sidebar-bg: #fafafa;
        --link: #000; --link-hover: #0070f3; --accent: #0070f3;
        --box-bg: #fff; --btn-bg: #000; --btn-fg: #fff;
    }
    :root[data-theme="dark"] {
        --bg: #000; --fg: #fff; --border: #333; --sidebar-bg: #000;
        --link: #fff; --link-hover: #58a6ff; --accent: #58a6ff;
        --box-bg: #111; --btn-bg: #fff; --btn-fg: #000;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: var(--font-main); background-color: var(--bg); color: var(--fg); margin: 0; font-size: 16px; line-height: 1.7; -webkit-font-smoothing: antialiased; }
    .page-container { display: grid; grid-template-columns: 240px 1fr; }
    .layout-homepage { grid-template-columns: 240px 1fr 1fr; }
    .sidebar { border-right: 1px solid var(--border); height: 100vh; position: sticky; top: 0; padding: 40px 20px; background-color: var(--sidebar-bg); }
    .nav-link { display: block; color: var(--link); text-decoration: none; padding: 8px 0; font-size: 14px; font-weight: 500; }
    .nav-title { font-size: 18px; font-weight: 700; }
    .nav-link:hover { color: var(--link-hover); }
    hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
    .theme-switcher { position: fixed; top: 20px; right: 20px; z-index: 1001; }
    .theme-btn { background: none; border: none; color: var(--link); cursor: pointer; padding: 5px; display: none; }
    .theme-btn.active-theme-btn { display: block; }
    :root[data-theme="light"] .theme-btn[data-theme="light"] { display: none; }
    :root[data-theme="light"] .theme-btn[data-theme="dark"] { display: block; }
    :root[data-theme="dark"] .theme-btn[data-theme="dark"] { display: none; }
    :root[data-theme="dark"] .theme-btn[data-theme="light"] { display: block; }
    .main-content { padding: 80px 60px; max-width: 800px; }
    .layout-docs .main-content { border-right: 1px solid var(--border); }
    h1 { font-size: 3em; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 20px; }
    p.intro-text { font-size: 1.1em; color: #888; max-width: 480px; }
    .homepage-links { display: flex; gap: 15px; margin-top: 30px; }
    .homepage-btn { padding: 10px 20px; background-color: var(--btn-bg); color: var(--btn-fg); text-decoration: none; border-radius: 5px; font-weight: 500; }
    .scratch { position: relative; display: inline-block; }
    .scratch::after { content: attr(data-text); position: absolute; left: 0; top: 0; color: var(--fg); text-decoration: line-through; transition: all 1s ease-in-out; }
    .scratch:hover::after { color: transparent; }
    .code-block { position: relative; margin-top: 2em; }
    .code-scroll-container { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
    .code-block pre { margin: 0; padding: 1.2em; font-size: 14px; }
    .copy-button { position: absolute; top: 12px; right: 12px; background: none; border: none; cursor: pointer; color: #888; }
    .copy-button:hover { color: var(--fg); }
    .feature-boxes { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 0; margin-top: 60px; }
    .feature-boxes .card { list-style: none; padding: 20px; border: 1px solid var(--border); border-radius: 8px; background-color: var(--box-bg); }
    .main-footer { grid-column: 1 / -1; padding: 20px; text-align: center; font-size: 14px; color: #888; border-top: 1px solid var(--border); }
    .main-footer a { color: #888; }
    .right-pane { padding: 20px; }
    #particle-canvas { width: 100%; height: 100%; min-height: 400px; }
    """
    with open(os.path.join(new_site_dir, 'css', 'style.css'), 'w') as f:
        f.write(css_content)
        
    # --- Create NEW JS file ---
    js_content = """
    document.addEventListener('DOMContentLoaded', () => {
        // --- THEME ---
        const themeButtons = document.querySelectorAll('.theme-btn');
        const htmlEl = document.documentElement;
        
        function setTheme(theme) {
            htmlEl.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
        }

        const savedTheme = localStorage.getItem('theme') || 'dark';
        setTheme(savedTheme);

        themeButtons.forEach(button => {
            button.addEventListener('click', () => {
                const newTheme = htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            });
        });

        // --- COPY BUTTON ---
        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', () => {
                const code = button.closest('.code-block').querySelector('pre').innerText;
                navigator.clipboard.writeText(code).then(() => {
                    button.title = 'Copied!';
                    setTimeout(() => { button.title = 'Copy code'; }, 2000);
                });
            });
        });

        // --- PARTICLE ANIMATION (Homepage Only) ---
        const canvas = document.getElementById('particle-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            let particles = [];
            
            function resizeCanvas() {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }

            class Particle {
                constructor(x, y) {
                    this.x = x; this.y = y;
                    this.vx = (Math.random() - 0.5) * 2;
                    this.vy = (Math.random() - 0.5) * 2;
                    this.size = Math.random() * 2 + 1;
                    this.color = `hsla(${Math.random() * 360}, 70%, 70%, 0.8)`;
                }
                update() {
                    this.x += this.vx;
                    this.y += this.vy;
                    if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                    if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
                }
                draw() {
                    ctx.fillStyle = this.color;
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            
            function init() {
                resizeCanvas();
                for (let i = 0; i < 50; i++) {
                    particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height));
                }
                animate();
            }

            function animate() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                particles.forEach(p => {
                    p.update();
                    p.draw();
                });
                requestAnimationFrame(animate);
            }
            
            window.addEventListener('resize', resizeCanvas);
            init();
        }
    });
    """
    with open(os.path.join(new_site_dir, 'js', 'main.js'), 'w') as f:
        f.write(js_content)

    print("Final website design has been generated in the 'site' directory.")

if __name__ == "__main__":
    apply_final_design()
