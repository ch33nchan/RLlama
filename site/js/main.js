document.addEventListener('DOMContentLoaded', () => {
    const sidebarLinks = document.querySelectorAll('.sidebar nav a');
    const currentPath = window.location.pathname.split('/').pop();

    sidebarLinks.forEach(link => {
        const linkPath = link.getAttribute('href').split('/').pop();
        link.classList.remove('active');
        // Handle index page case
        if (currentPath === linkPath || (currentPath === '' && linkPath === 'index.html')) {
            link.classList.add('active');
        }
    });
});