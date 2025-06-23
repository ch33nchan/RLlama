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

    # --- Get a list of all pages for the new sidebar ---
    all_pages = sorted([f for f in os.listdir(backup_dir) if f.endswith('.html')])
    
    # --- Pygments CSS ---
    pygments_css = HtmlFormatter(style='one-dark').get_style_defs('.code-container')

    for filename in all_pages:
        if not os.path.exists(os.path.join(backup_dir, filename)):
            continue
            
        with open(os.path.join(backup_dir, filename), 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        old_content = soup.find('main', class_='content') or soup.body
        
        if old_content:
            for tag in old_content.find_all(['nav', 'div'], class_=['sidebar', 'theme-switch']):
                tag.decompose()

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
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="theme-switcher">
        <button class="theme-btn" data-theme="light" title="Light Theme"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg></button>
        <button class="theme-btn" data-theme="dark" title="Dark Theme"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg></button>
    </div>
    <div class="page-container">
        <aside class="sidebar">
            <nav id="new-sidebar-nav"></nav>
        </aside>
        <main class="main-content">
            <header id="page-header"></header>
            <div id="content-container"></div>
        </main>
        <aside class="right-pane">
            <canvas id="physics-canvas"></canvas>
        </aside>
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
        sidebar_nav.append(BeautifulSoup('<a href="https://github.com/ch33nchan/RLlama/" target="_blank" class="nav-link">GitHub</a>', 'html.parser'))
        sidebar_nav.append(BeautifulSoup('<a href="https://pypi.org/project/rllama/" target="_blank" class="nav-link">PyPI</a>', 'html.parser'))
        sidebar_nav.append(BeautifulSoup('<hr/>', 'html.parser'))
        for page in all_pages:
            if page != 'index.html':
                page_name = page.replace('.html', '').replace('_', ' ').capitalize()
                sidebar_nav.append(BeautifulSoup(f'<a href="{page}" class="nav-link">{page_name}</a>', 'html.parser'))
        
        # --- Inject Content and Header ---
        new_soup.find(id='content-container').append(old_content)
        header = new_soup.find(id='page-header')
        if filename == 'index.html':
            header.append(BeautifulSoup('<h1><span class="scratch" data-text="Struggling with rewards?">RLlama is here to help.</span></h1>', 'html.parser'))
            # Find and wrap feature cards in boxes
            if old_content:
                features_section = old_content.find('section', class_='features')
                if features_section:
                    features_section['class'] = features_section.get('class', []) + ['feature-boxes']

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
                highlighted_code = highlight(code_content, PythonLexer(), HtmlFormatter())
                
                new_code_container = BeautifulSoup(f'<div class="code-container">{highlighted_code}</div>', 'html.parser')
                new_code_container.find('div').append(BeautifulSoup('<button class="copy-button" title="Copy code"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></button>', 'html.parser'))
                
                pre_tag.replace_with(new_code_container)
        
        with open(os.path.join(new_site_dir, filename), 'w') as f:
            f.write(str(new_soup.prettify()))
    
    # --- Create NEW CSS file ---
    css_content = pygments_css + """
    :root {
        --font-main: 'Space Mono', monospace;
    }
    :root[data-theme="light"] {
        --bg-color: #e9e9e9; --fg-color: #1a1a1a; --border-color: #d4d4d4;
        --sidebar-bg: #e9e9e9;
        --link-color: #333; --link-hover-color: #000;
        --accent: #0070f3;
        --box-bg: #f5f5f5;
    }
    :root[data-theme="dark"] {
        --bg-color: #000; --fg-color: #e6e6e6; --border-color: #2a2a2a;
        --sidebar-bg: #000;
        --link-color: #ccc; --link-hover-color: #fff;
        --accent: #58a6ff;
        --box-bg: #111;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
        font-family: var(--font-main); background-color: var(--bg-color); color: var(--fg-color);
        margin: 0; font-size: 16px; line-height: 1.6;
    }
    .page-container { display: grid; grid-template-columns: 240px 1fr 240px; min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--border-color); height: 100vh; position: sticky; top: 0; display: flex; flex-direction: column; padding: 40px 20px; }
    .sidebar nav { flex-grow: 1; }
    .nav-link { display: block; color: var(--link-color); text-decoration: none; padding: 8px 0; font-size: 14px; }
    .nav-title { font-size: 18px; font-weight: 700; }
    .nav-link:hover { color: var(--link-hover-color); }
    hr { border: none; border-top: 1px solid var(--border-color); margin: 10px 0; }
    .theme-switcher { position: fixed; top: 20px; right: 20px; z-index: 1001; }
    .theme-btn { background: none; border: none; color: var(--link-color); cursor: pointer; padding: 5px; }
    .theme-btn.active { color: var(--accent); }
    .main-content { padding: 40px 60px; max-width: 800px; border-right: 1px solid var(--border-color); }
    h1, h2, h3 { margin-top: 1.5em; margin-bottom: 0.5em; font-weight: 400; }
    h1 { font-size: 2.2em; }
    .scratch { position: relative; display: inline-block; }
    .scratch::after { content: attr(data-text); position: absolute; left: 0; top: 0; color: var(--fg-color); text-decoration: line-through; transition: all 1s ease-in-out; }
    .scratch:hover::after { color: transparent; }
    .code-container { position: relative; margin-top: 2em; }
    .code-container .highlight { background-color: var(--box-bg); border: 1px solid var(--border-color); border-radius: 4px; }
    .copy-button { position: absolute; top: 10px; right: 10px; background: none; border: none; cursor: pointer; color: #888; }
    .copy-button:hover { color: var(--fg-color); }
    .feature-boxes { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 0; }
    .feature-boxes .card { list-style: none; padding: 20px; border: 1px solid var(--border-color); background-color: var(--box-bg); }
    .main-footer { grid-column: 1 / -1; padding: 20px; text-align: center; font-size: 14px; color: #888; border-top: 1px solid var(--border-color); }
    .main-footer a { color: #888; }
    .right-pane { padding: 20px; }
    #physics-canvas { width: 100%; height: 100%; }
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
            themeButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.theme-btn[data-theme="${theme}"]`).classList.add('active');
        }

        const savedTheme = localStorage.getItem('theme') || 'dark'; // Dark by default
        setTheme(savedTheme);

        themeButtons.forEach(button => {
            button.addEventListener('click', () => setTheme(button.dataset.theme));
        });

        // --- COPY BUTTON ---
        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', () => {
                const code = button.closest('.code-container').querySelector('pre').innerText;
                navigator.clipboard.writeText(code).then(() => {
                    button.title = 'Copied!';
                    setTimeout(() => { button.title = 'Copy code'; }, 2000);
                });
            });
        });

        // --- BALL ANIMATION ---
        const canvas = document.getElementById('physics-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            let ball = { x: 100, y: 100, vx: 5, vy: 2, radius: 25 };
            const color = getComputedStyle(document.documentElement).getPropertyValue('--fg-color');

            function draw() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.fill();

                if (ball.x + ball.vx > canvas.width || ball.x + ball.vx < 0) ball.vx = -ball.vx;
                if (ball.y + ball.vy > canvas.height || ball.y + ball.vy < 0) ball.vy = -ball.vy;
                
                ball.x += ball.vx;
                ball.y += ball.vy;
                
                requestAnimationFrame(draw);
            }
            draw();
        }
    });
    """
    with open(os.path.join(new_site_dir, 'js', 'main.js'), 'w') as f:
        f.write(js_content)

    print("Final website design has been generated in the 'site' directory.")

if __name__ == "__main__":
    apply_final_design()
