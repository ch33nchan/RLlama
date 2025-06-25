// app.js - RLlama Documentation Website JavaScript with TE Aesthetics

// ============================================================================
// Global Variables and Configuration
// ============================================================================

let currentTheme = 'light';
let isMenuOpen = false;
let activeTab = 'all';
let scrollPosition = 0;
let loadingComplete = false;

// TE Animation configuration
const teConfig = {
    duration: 200,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    stagger: 50,
    gridSize: 8,
    dotSize: 2
};

// Component data for matrix
const componentData = {
    basic: [
        { name: 'LengthReward', description: 'Text/sequence length optimization with configurable targets' },
        { name: 'ConstantReward', description: 'Baseline rewards for testing and comparison' },
        { name: 'ThresholdReward', description: 'Binary threshold-based rewards with conditions' },
        { name: 'RangeReward', description: 'Range-based reward functions with smooth transitions' },
        { name: 'ProportionalReward', description: 'Linear scaling rewards with configurable factors' },
        { name: 'BinaryReward', description: 'Simple binary conditions for quick decisions' }
    ],
    llm: [
        { name: 'PerplexityReward', description: 'Language model fluency assessment via perplexity' },
        { name: 'SemanticSimilarityReward', description: 'Embedding-based similarity with multiple metrics' },
        { name: 'ToxicityReward', description: 'Safety and toxicity detection with penalties' },
        { name: 'FactualityReward', description: 'Factual accuracy verification with claim checking' },
        { name: 'CreativityReward', description: 'Multi-dimensional creativity assessment' }
    ],
    learning: [
        { name: 'MetaLearningReward', description: 'MAML-style adaptation for task performance' },
        { name: 'UncertaintyBasedReward', description: 'Uncertainty estimates for exploration' },
        { name: 'AdversarialReward', description: 'Adversarial training for robust functions' },
        { name: 'AdaptiveClippingReward', description: 'Dynamic reward clipping based on statistics' },
        { name: 'GradualCurriculumReward', description: 'Progressive difficulty increase' },
        { name: 'HindsightExperienceReward', description: 'HER for sparse reward environments' }
    ],
    robotics: [
        { name: 'CollisionAvoidanceReward', description: 'Safety navigation with margin enforcement' },
        { name: 'EnergyEfficiencyReward', description: 'Power optimization for robot behavior' },
        { name: 'TaskCompletionReward', description: 'Goal achievement with progress tracking' },
        { name: 'SmoothTrajectoryReward', description: 'Natural motion with jerk penalties' }
    ],
    advanced: [
        { name: 'MultiObjectiveReward', description: 'Pareto optimization for competing objectives' },
        { name: 'HierarchicalReward', description: 'Goal decomposition with hierarchy' },
        { name: 'ContrastiveReward', description: 'Contrastive learning principles' },
        { name: 'TemporalConsistencyReward', description: 'Sequence consistency enforcement' }
    ]
};

// ============================================================================
// DOM Ready and Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    initializeLoadingScreen();
    initializeHeroCanvas();
    initializeComponentMatrix();
    setupTerminalAnimation();
    initializeCounters();
    setupScrollEffects();
    initializeCopyButtons();
    setupTabSystem();
    setupNavigation();
});

// ============================================================================
// Core Initialization Functions
// ============================================================================

function initializeApp() {
    console.log('🦙 RLlama TE Documentation Website Initialized');
    
    // Add loaded class after initialization
    setTimeout(() => {
        document.body.classList.add('loaded');
        loadingComplete = true;
    }, 100);
}

function setupEventListeners() {
    // Navigation toggle
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', toggleMobileMenu);
    }
    
    // Scroll events
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize, { passive: true });
    
    // Keyboard navigation
    setupKeyboardNavigation();
}

// ============================================================================
// Loading Screen with TE Animation
// ============================================================================

function initializeLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    const loadingProgress = document.querySelector('.loading-progress');
    
    if (!loadingScreen || !loadingProgress) return;
    
    // Animate loading progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
            progress = 100;
            clearInterval(progressInterval);
            
            // Hide loading screen
            setTimeout(() => {
                loadingScreen.classList.add('fade-out');
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 500);
            }, 500);
        }
        loadingProgress.style.width = `${progress}%`;
    }, 100);
}

// ============================================================================
// Hero Canvas Animation
// ============================================================================

function initializeHeroCanvas() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Particle system
    const particles = [];
    const numParticles = 30;
    
    // Create particles
    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.2
        });
    }
    
    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update and draw particles
        particles.forEach(particle => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Wrap around edges
            if (particle.x < 0) particle.x = canvas.width;
            if (particle.x > canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = canvas.height;
            if (particle.y > canvas.height) particle.y = 0;
            
            // Draw particle
            ctx.fillStyle = `rgba(0, 0, 0, ${particle.opacity})`;
            ctx.fillRect(particle.x, particle.y, particle.size, particle.size);
        });
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// ============================================================================
// Component Matrix System
// ============================================================================

function initializeComponentMatrix() {
    const matrixGrid = document.getElementById('component-matrix');
    const matrixButtons = document.querySelectorAll('.matrix-btn');
    
    if (!matrixGrid) return;
    
    // Setup matrix controls
    matrixButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const category = btn.getAttribute('data-category');
            switchMatrixCategory(category, matrixButtons);
        });
    });
    
    // Initial render
    renderComponentMatrix('all');
}

function switchMatrixCategory(category, buttons) {
    // Update active button
    buttons.forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.querySelector(`[data-category="${category}"]`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Update matrix
    renderComponentMatrix(category);
    activeTab = category;
}

function renderComponentMatrix(category) {
    const matrixGrid = document.getElementById('component-matrix');
    if (!matrixGrid) return;
    
    // Get components for category
    let components = [];
    if (category === 'all') {
        Object.values(componentData).forEach(categoryComponents => {
            components = components.concat(categoryComponents);
        });
    } else {
        components = componentData[category] || [];
    }
    
    // Clear existing items
    matrixGrid.innerHTML = '';
    
    // Create matrix items
    components.forEach((component, index) => {
        const item = document.createElement('div');
        item.className = 'matrix-item';
        item.innerHTML = `
            <h4 class="mono-text">${component.name}</h4>
            <p class="mono-text">${component.description}</p>
        `;
        
        // Add staggered animation
        item.style.opacity = '0';
        item.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            item.style.transition = `opacity ${teConfig.duration}ms ${teConfig.easing}, transform ${teConfig.duration}ms ${teConfig.easing}`;
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, index * teConfig.stagger);
        
        matrixGrid.appendChild(item);
    });
}

// ============================================================================
// Terminal Animation
// ============================================================================

function setupTerminalAnimation() {
    const codeAnimation = document.getElementById('code-animation');
    if (!codeAnimation) return;
    
    const codeLines = [
        'from rllama import RewardEngine',
        '',
        '# Initialize engine',
        'engine = RewardEngine("config.yaml")',
        '',
        '# Compute reward',
        'context = {',
        '    "response": "Hello, world!",',
        '    "query": "Say hello"',
        '}',
        '',
        'reward = engine.compute(context)',
        'print(f"Reward: {reward:.4f}")',
        '',
        '# 🦙 RLlama: Making RL simple!'
    ];
    
    let currentLine = 0;
    let currentChar = 0;
    
    function typeNextChar() {
        if (currentLine >= codeLines.length) {
            // Restart animation after delay
            setTimeout(() => {
                codeAnimation.innerHTML = '';
                currentLine = 0;
                currentChar = 0;
                typeNextChar();
            }, 3000);
            return;
        }
        
        const line = codeLines[currentLine];
        
        if (currentChar <= line.length) {
            const displayText = codeLines.slice(0, currentLine).join('\n') + 
                               (currentLine > 0 ? '\n' : '') + 
                               line.substring(0, currentChar) + 
                               (currentChar < line.length ? '█' : '');
            
            codeAnimation.innerHTML = displayText;
            currentChar++;
            
            setTimeout(typeNextChar, 50);
        } else {
            currentLine++;
            currentChar = 0;
            setTimeout(typeNextChar, 200);
        }
    }
    
    // Start animation after delay
    setTimeout(typeNextChar, 1000);
}

// ============================================================================
// Counter Animations
// ============================================================================

function initializeCounters() {
    const counters = document.querySelectorAll('.stat-number[data-target]');
    
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                animateCounter(entry.target);
                entry.target.classList.add('counted');
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-target'));
    const duration = 1500;
    const start = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(easeOut * target);
        
        element.textContent = current.toString().padStart(2, '0');
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target.toString().padStart(2, '0');
        }
    }
    
    requestAnimationFrame(updateCounter);
}

// ============================================================================
// Navigation and Menu Functions
// ============================================================================

function toggleMobileMenu() {
    const navMenu = document.getElementById('nav-menu');
    const navToggle = document.getElementById('nav-toggle');
    
    isMenuOpen = !isMenuOpen;
    
    navMenu.classList.toggle('active', isMenuOpen);
    navToggle.classList.toggle('active', isMenuOpen);
    
    // Animate hamburger menu
    animateHamburgerMenu(navToggle, isMenuOpen);
    
    // Prevent body scroll when menu is open
    document.body.style.overflow = isMenuOpen ? 'hidden' : '';
}

function animateHamburgerMenu(toggle, isOpen) {
    const spans = toggle.querySelectorAll('span');
    
    if (isOpen) {
        spans[0].style.transform = 'rotate(45deg) translate(3px, 3px)';
        spans[1].style.opacity = '0';
        spans[2].style.transform = 'rotate(-45deg) translate(3px, -3px)';
    } else {
        spans[0].style.transform = 'none';
        spans[1].style.opacity = '1';
        spans[2].style.transform = 'none';
    }
}

function setupNavigation() {
    // Smooth scroll for navigation links
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                smoothScrollTo(targetElement);
                
                // Close mobile menu if open
                if (isMenuOpen) {
                    toggleMobileMenu();
                }
                
                // Update active nav item
                updateActiveNavItem(link);
            }
        });
    });
}

function updateActiveNavItem(activeLink) {
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Add active class to clicked link
    activeLink.classList.add('active');
}

// ============================================================================
// Scroll Handling and Effects
// ============================================================================

function handleScroll() {
    const currentScroll = window.pageYOffset;
    scrollPosition = currentScroll;
    
    // Update navbar background
    updateNavbarBackground(currentScroll);
    
    // Update active section in navigation
    updateActiveSection();
    
    // Handle scroll-triggered animations
    handleScrollAnimations();
}

function updateNavbarBackground(scrollY) {
    const navbar = document.getElementById('navbar');
    
    if (scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
}

function updateActiveSection() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.offsetHeight;
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });
    
    // Update navigation
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

function handleScrollAnimations() {
    const animatedElements = document.querySelectorAll('.fade-in, .step-card, .feature-card, .doc-section');
    
    animatedElements.forEach(element => {
        if (isElementInViewport(element) && !element.classList.contains('animated')) {
            element.classList.add('fade-in');
            element.classList.add('animated');
        }
    });
}

function setupScrollEffects() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '50px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                entry.target.classList.add('fade-in');
                entry.target.classList.add('animated');
            }
        });
    }, observerOptions);
    
    // Observe elements
    const elementsToObserve = document.querySelectorAll('.step-card, .feature-card, .doc-section, .cookbook-card, .api-item');
    elementsToObserve.forEach(element => observer.observe(element));
}

// ============================================================================
// Tab System
// ============================================================================

function setupTabSystem() {
    setupExampleTabs();
}

function setupExampleTabs() {
    const exampleTabButtons = document.querySelectorAll('.examples-tabs .tab-btn');
    const exampleTabPanes = document.querySelectorAll('.examples .example-pane');
    
    exampleTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            switchExampleTab(targetTab, exampleTabButtons, exampleTabPanes);
        });
    });
}

function switchExampleTab(targetTab, buttons, panes) {
    // Remove active class from all buttons and panes
    buttons.forEach(btn => btn.classList.remove('active'));
    panes.forEach(pane => pane.classList.remove('active'));
    
    // Add active class to target button and pane
    const targetButton = document.querySelector(`.examples-tabs .tab-btn[data-tab="${targetTab}"]`);
    const targetPane = document.getElementById(targetTab);
    
    if (targetButton && targetPane) {
        targetButton.classList.add('active');
        targetPane.classList.add('active');
        
        // Animate tab content
        animateTabContent(targetPane);
    }
}

function animateTabContent(pane) {
    const content = pane.querySelector('.example-container');
    
    if (content) {
        content.style.opacity = '0';
        content.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            content.style.transition = `opacity ${teConfig.duration}ms ${teConfig.easing}, transform ${teConfig.duration}ms ${teConfig.easing}`;
            content.style.opacity = '1';
            content.style.transform = 'translateY(0)';
        }, 50);
    }
}

// ============================================================================
// Copy Functionality
// ============================================================================

function initializeCopyButtons() {
    setupCopyButtons();
    addCopyButtonsToCodeBlocks();
}

function setupCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn[data-copy]');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            const textToCopy = button.getAttribute('data-copy');
            copyToClipboard(textToCopy, button);
        });
    });
}

function addCopyButtonsToCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const preElement = block.parentElement;
        if (preElement.querySelector('.copy-btn')) return; // Already has copy button
        
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.setAttribute('aria-label', 'Copy code');
        
        copyButton.addEventListener('click', () => {
            const code = block.textContent;
            copyToClipboard(code, copyButton);
        });
        
        preElement.style.position = 'relative';
        preElement.appendChild(copyButton);
    });
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        // Show success feedback
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.classList.add('copied');
        
        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.classList.remove('copied');
        }, 2000);
        
        // Show toast notification
        showToast('Code copied to clipboard!', 'success');
    }).catch(() => {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        showToast('Code copied to clipboard!', 'success');
    });
}

// ============================================================================
// Smooth Scrolling
// ============================================================================

function smoothScrollTo(element) {
    const targetPosition = element.offsetTop - 80; // Account for fixed navbar
    const startPosition = window.pageYOffset;
    const distance = targetPosition - startPosition;
    const duration = 600;
    let start = null;
    
    function animation(currentTime) {
        if (start === null) start = currentTime;
        const timeElapsed = currentTime - start;
        const run = easeInOutQuad(timeElapsed, startPosition, distance, duration);
        window.scrollTo(0, run);
        
        if (timeElapsed < duration) {
            requestAnimationFrame(animation);
        }
    }
    
    requestAnimationFrame(animation);
}

function easeInOutQuad(t, b, c, d) {
    t /= d / 2;
    if (t < 1) return c / 2 * t * t + b;
    t--;
    return -c / 2 * (t * (t - 2) - 1) + b;
}

// ============================================================================
// Utility Functions
// ============================================================================

function isElementInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function handleResize() {
    // Handle responsive changes
    if (window.innerWidth > 768 && isMenuOpen) {
        toggleMobileMenu();
    }
}

function setupKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        // Escape key closes mobile menu
        if (e.key === 'Escape' && isMenuOpen) {
            toggleMobileMenu();
        }
        
        // Arrow keys for matrix navigation
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            const activeMatrixButton = document.querySelector('.matrix-btn.active');
            if (activeMatrixButton) {
                const matrixButtons = Array.from(document.querySelectorAll('.matrix-btn'));
                const currentIndex = matrixButtons.indexOf(activeMatrixButton);
                
                let newIndex;
                if (e.key === 'ArrowLeft') {
                    newIndex = currentIndex > 0 ? currentIndex - 1 : matrixButtons.length - 1;
                } else {
                    newIndex = currentIndex < matrixButtons.length - 1 ? currentIndex + 1 : 0;
                }
                
                matrixButtons[newIndex].click();
                matrixButtons[newIndex].focus();
            }
        }
    });
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} mono-text`;
    toast.textContent = message;
    
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 24px;
        background: ${type === 'success' ? '#00ff66' : type === 'error' ? '#ff0066' : '#0066ff'};
        color: #000000;
        border: 1px solid #000000;
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
    }, 10);
    
    setTimeout(() => {
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 200);
    }, 3000);
}

// ============================================================================
// TE-specific Animations
// ============================================================================

function initializeTEAnimations() {
    // Animate TE dots on hover
    const teDots = document.querySelectorAll('.te-dot, .brand-dot, .icon-dot');
    
    teDots.forEach(dot => {
        dot.addEventListener('mouseenter', () => {
            dot.style.transform = 'scale(1.5)';
            dot.style.backgroundColor = '#ff6600';
        });
        
        dot.addEventListener('mouseleave', () => {
            dot.style.transform = 'scale(1)';
            dot.style.backgroundColor = '';
        });
    });
    
    // Matrix item hover effects
    const matrixItems = document.querySelectorAll('.matrix-item');
    
    matrixItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.transform = 'translateY(-2px)';
        });
        
        item.addEventListener('mouseleave', () => {
            item.style.transform = 'translateY(0)';
        });
    });
}

// ============================================================================
// Export functions for external use
// ============================================================================

window.RLlamaApp = {
    smoothScrollTo,
    showToast,
    switchMatrixCategory,
    renderComponentMatrix,
    copyToClipboard,
    animateCounter
};

// Initialize TE-specific animations after DOM load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initializeTEAnimations, 500);
});

console.log('🦙 RLlama TE App.js loaded successfully!');
