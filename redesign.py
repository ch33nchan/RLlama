import os
import shutil

# Create the site directory
if not os.path.exists('site'):
    os.makedirs('site')
if not os.path.exists('site/css'):
    os.makedirs('site/css')
if not os.path.exists('site/js'):
    os.makedirs('site/js')

# Backup the old docs directory
if os.path.exists('docs'):
    if os.path.exists('docs_backup'):
        shutil.rmtree('docs_backup')
    shutil.move('docs', 'docs_backup')


# --- Create the new HTML file ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLlama</title>
    <link rel="stylesheet" href="css/teeny.css">
</head>
<body>
    <div class="sidebar">
        <button id="sidebar-toggle">☰</button>
        <nav>
            <a href="https://github.com/ch33nchan/RLlama/" target="_blank">GitHub</a>
            <a href="https://pypi.org/project/rllama/" target="_blank">PyPI</a>
        </nav>
    </div>
    <div class="main-content">
        <header>
            <h1><span class="scratch" data-text="Reward Engineering is Boring.">RLlama makes it fun.</span></h1>
        </header>
        <div class="animation-placeholder">
            </div>
        <pre><code class="language-python">
# Your code here
def hello_rllama():
    print("Hello, RLlama!")
        </code></pre>
        <button class="copy-button">Copy</button>
    </div>
    <footer>
        <p>engineered and designed by <a href="https://srinivastb.netlify.app/" target="_blank">srini</a> with <3</p>
    </footer>
    <script src="js/teeny.js"></script>
</body>
</html>
"""

with open('site/index.html', 'w') as f:
    f.write(html_content)

# --- Create the new CSS file ---
css_content = """
body {
    font-family: monospace;
    background-color: #1a1a1a;
    color: #f0f0f0;
    margin: 0;
    padding: 0;
    display: flex;
}

.sidebar {
    width: 200px;
    height: 100vh;
    background-color: #111;
    padding: 20px;
    position: fixed;
    left: -200px;
    transition: left 0.3s ease;
}

.sidebar.active {
    left: 0;
}

#sidebar-toggle {
    position: fixed;
    top: 20px;
    left: 20px;
    background: none;
    border: none;
    color: #f0f0f0;
    font-size: 24px;
    cursor: pointer;
}

.main-content {
    margin-left: 20px;
    padding: 20px;
    width: 100%;
}

header h1 {
    font-size: 3em;
}

.scratch {
    position: relative;
    display: inline-block;
}

.scratch::after {
    content: attr(data-text);
    position: absolute;
    left: 0;
    top: 0;
    color: #f0f0f0;
    text-decoration: line-through;
    transition: all 2s ease;
}

.scratch:hover::after {
    color: transparent;
}


pre {
    background-color: #2d2d2d;
    padding: 20px;
    border-radius: 5px;
    position: relative;
}

.copy-button {
    position: absolute;
    top: 10px;
    right: 10px;
    background-color: #444;
    color: #f0f0f0;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
}

code .keyword { color: #569cd6; }
code .string { color: #ce9178; }
code .comment { color: #6a9955; }
code .function { color: #dcdcaa; }
code .number { color: #b5cea8; }


footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 0.8em;
}

footer a {
    color: #f0f0f0;
    text-decoration: none;
}
"""

with open('site/css/teeny.css', 'w') as f:
    f.write(css_content)

# --- Create the new JavaScript file ---
js_content = """
document.addEventListener('DOMContentLoaded', (event) => {
    const sidebar = document.querySelector('.sidebar');
    const sidebarToggle = document.querySelector('#sidebar-toggle');

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });

    const copyButton = document.querySelector('.copy-button');
    copyButton.addEventListener('click', () => {
        const code = document.querySelector('code').innerText;
        navigator.clipboard.writeText(code).then(() => {
            copyButton.innerText = 'Copied!';
            setTimeout(() => {
                copyButton.innerText = 'Copy';
            }, 2000);
        });
    });
});
"""

with open('site/js/teeny.js', 'w') as f:
    f.write(js_content)

print("Website redesign complete. New files are in the 'site' directory.")
