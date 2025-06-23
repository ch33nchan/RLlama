document.addEventListener('DOMContentLoaded', function() {
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.querySelector('.sidebar');
  const mainContent = document.querySelector('.main-content');
  const rightPane = document.querySelector('.right-pane');
  
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', function() {
      // On mobile, use show/hide instead of collapse
      if (window.innerWidth <= 768) {
        sidebar.classList.toggle('show');
      } else {
        sidebar.classList.toggle('collapsed');
        document.body.style.gridTemplateColumns = 
          sidebar.classList.contains('collapsed') ? '0 1fr' : 'auto 1fr';
        if (mainContent) {
          mainContent.classList.toggle('sidebar-collapsed');
        }
        if (rightPane) {
          rightPane.classList.toggle('sidebar-collapsed');
        }
      }
    });
  }
  
  // Handle window resize
  window.addEventListener('resize', function() {
    if (window.innerWidth > 768) {
      sidebar.classList.remove('show');
    }
  });
});
