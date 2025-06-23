import os
import shutil
from bs4 import BeautifulSoup

def apply_new_design():
    # --- Configuration ---
    backup_dir = 'docs_backup'
    new_site_dir = 'site'
    
    # --- Clean up old 'site' directory ---
    if os.path.exists(new_site_dir):
        shutil.rmtree(new_site_dir)
    os.makedirs(os.path.join(new_site_dir, 'css'))
    os.makedirs(os.path.join(new_site_dir, 'js'))

    # --- Process each HTML file from the backup ---
    for filename in os.listdir(backup_dir):
        if filename.endswith('.html'):
            with open(os.path.join(backup_dir, filename), 'r') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Extract the core content from the old site
            old_content = soup.find('main', class_='content') or soup.body

            # Create the new page structure
            new_soup = BeautifulSoup(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLlama - {filename.replace('.html', '').capitalize()}</title>
    <link rel="stylesheet" href="css/teeny.css">
</head>
<body>
    <div class="sidebar">
        <button id="sidebar-toggle">☰</button>
        <nav>
            <a href="index.html">Home</a>
            <a href="https://github.com/ch33nchan/RLlama/" target="_blank">GitHub</a>
            <a href="https://pypi.org/project/rllama/" target="_blank">PyPI</a>
        </nav>
    </div>
    <div class="main-content">
        <header id="page-header">
        </header>
        <div id="content-container">
        </div>
    </div>
    <footer>
        <p>engineered and designed by <a href="https://srinivastb.netlify.app/" target="_blank">srini</a> with <3</p>
    </footer>
    <script src="js/teeny.js"></script>
</body>
</html>
            """, 'html.parser')

            # Inject the old content into the new structure
            new_soup.find(id='content-container').append(old_content)
            
            # Set the header based on the page
            header = new_soup.find(id='page-header')
            if filename == 'index.html':
                header.append(BeautifulSoup('<h1><span class="scratch" data-text="Reward Engineering is Boring.">RLlama makes it fun.</span></h1><div class="knob-animation"><div class="knob"></div></div>', 'html.parser'))
            else:
                header.append(BeautifulSoup(f"<h1>{filename.replace('.html', '').replace('_', ' ').capitalize()}</h1>", 'html.parser'))

            # Restyle code blocks
            for code_block in new_soup.find_all('pre'):
                code_block.wrap(new_soup.new_tag("div", **{'class': 'code-container'}))
                code_block.find_parent('div', class_='code-container').append(BeautifulSoup('<button class="copy-button">Copy</button>', 'html.parser'))


            # Write the new, styled HTML file
            with open(os.path.join(new_site_dir, filename), 'w') as f:
                f.write(str(new_soup))
    
    # --- Create CSS file ---
    css_content = """
    body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; background-color: #0d0d0d; color: #e6e6e6; margin: 0; padding: 0; display: flex; text-align: left; }
    .sidebar { width: 200px; height: 100vh; background-color: #1a1a1a; padding: 20px; position: fixed; transform: translateX(-220px); transition: transform 0.3s ease-in-out; border-right: 1px solid #333; }
    .sidebar.active { transform: translateX(0); }
    #sidebar-toggle { position: fixed; top: 20px; left: 20px; background: none; border: none; color: #e6e6e6; font-size: 24px; cursor: pointer; z-index: 1001; }
    .sidebar nav { margin-top: 60px; }
    .sidebar nav a { display: block; color: #e6e6e6; text-decoration: none; padding: 10px 0; border-bottom: 1px solid #333; }
    .main-content { margin-left: 20px; padding: 40px; width: 100%; }
    header h1 { font-size: 2.5em; font-weight: 500; margin-bottom: 40px; }
    .scratch { position: relative; display: inline-block; }
    .scratch::after { content: attr(data-text); position: absolute; left: 0; top: 0; color: #e6e6e6; text-decoration: line-through; transition: all 1s ease-in-out; }
    .scratch:hover::after { color: transparent; }
    .code-container { position: relative; margin-bottom: 20px; }
    pre { background-color: #1e1e1e; padding: 20px; border-radius: 5px; border: 1px solid #333; overflow-x: auto; }
    code { color: #d4d4d4; font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; }
    code span.kw { color: #569cd6; } /* Keywords */
    code span.st, code span.ss { color: #ce9178; } /* Strings */
    code span.co { color: #6a9955; font-style: italic; } /* Comments */
    code span.op { color: #d4d4d4; } /* Operators */
    code span.dv, code span.fl, code span.cn { color: #b5cea8; } /* Numbers, constants */
    .copy-button { position: absolute; top: 10px; right: 10px; background-color: #333; color: #e6e6e6; border: 1px solid #555; padding: 5px 10px; border-radius: 3px; cursor: pointer; opacity: 0.5; transition: opacity 0.2s; }
    .code-container:hover .copy-button { opacity: 1; }
    .copy-button:active { background-color: #444; }
    footer { position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; font-size: 0.8em; color: #555; }
    footer a { color: #888; text-decoration: none; }
    .knob-animation { margin: 40px 0; }
    .knob { width: 50px; height: 50px; background-color: #2a2a2a; border-radius: 50%; border: 2px solid #444; position: relative; animation: rotateKnob 5s linear infinite; }
    .knob::after { content: ''; width: 4px; height: 15px; background-color: #e6e6e6; position: absolute; top: 5px; left: 50%; transform: translateX(-50%); }
    @keyframes rotateKnob { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    """
    with open(os.path.join(new_site_dir, 'css', 'teeny.css'), 'w') as f:
        f.write(css_content)
        
    # --- Create JS file ---
    js_content = """
    document.addEventListener('DOMContentLoaded', () => {
        const sidebar = document.querySelector('.sidebar');
        const sidebarToggle = document.querySelector('#sidebar-toggle');
        sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('active'));

        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', () => {
                const code = button.previousElementSibling.querySelector('code').innerText;
                navigator.clipboard.writeText(code).then(() => {
                    button.innerText = 'Copied!';
                    setTimeout(() => { button.innerText = 'Copy'; }, 2000);
                });
            });
        });
    });
    """
    with open(os.path.join(new_site_dir, 'js', 'teeny.js'), 'w') as f:
        f.write(js_content)

    print("Website redesign and content migration complete. New files are in the 'site' directory.")
    print("Run 'open site/index.html' to see the new site.")

# --- Run the main function ---
if __name__ == "__main__":
    apply_new_design()
