// app.js - Complete RLlama Documentation Script
console.log('🦙 RLlama Documentation Loading...');

// Global variables
let currentVersion = 'v0.7.0';
let isMenuOpen = false;
let activeTab = 'all';
let scrollPosition = 0;
let loadingComplete = false;

// Real RLlama component data
const componentData = {
    basic: [
        { 
            name: 'LengthReward', 
            description: 'Rewards responses based on their length relative to a target length with configurable tolerance',
            code: `from rllama.rewards.components import LengthReward

# Configure length-based reward
length_reward = LengthReward(
    target_length=150,
    strength=1.0,
    tolerance=0.1
)

context = {
    "response": "This is a sample response for length evaluation.",
    "query": "Generate a response"
}

reward = length_reward.calculate(context)
print(f"Length reward: {reward:.4f}")`
        },
        { 
            name: 'DiversityReward', 
            description: 'Encourages diverse outputs by tracking response history and penalizing repetitive responses',
            code: `from rllama.rewards.components import DiversityReward

diversity_reward = DiversityReward(
    history_size=10,
    similarity_threshold=0.8,
    diversity_weight=1.0
)

context = {
    "response": "Novel creative response with unique ideas",
    "query": "Be creative",
    "response_history": ["previous", "responses", "list"]
}

reward = diversity_reward.calculate(context)
print(f"Diversity reward: {reward:.4f}")`
        }
    ],
    models: [
        {
            name: 'MLPRewardModel',
            description: 'Multi-layer perceptron for learning complex reward functions',
            code: `from rllama.models import MLPRewardModel
import torch

model = MLPRewardModel(
    input_dim=768,
    hidden_dims=[256, 128, 64],
    activation=torch.nn.ReLU
)

state = torch.randn(1, 768)
reward = model(state)
print(f"Predicted reward: {reward.item():.4f}")`
        },
        {
            name: 'EnsembleRewardModel',
            description: 'Ensemble of reward models with uncertainty estimates',
            code: `from rllama.models import EnsembleRewardModel
import torch

model = EnsembleRewardModel(
    input_dim=768,
    hidden_dims=[256, 128],
    num_models=5
)

state = torch.randn(1, 768)
reward, uncertainty = model(state, return_uncertainty=True)
print(f"Reward: {reward.item():.4f}")
print(f"Uncertainty: {uncertainty.item():.4f}")`
        }
    ],
    rlhf: [
        {
            name: 'PreferenceCollector',
            description: 'Collects and manages human preference data for RLHF training',
            code: `from rllama.rlhf import PreferenceCollector
import numpy as np

collector = PreferenceCollector(buffer_size=10000)

state_a = np.random.randn(4)
state_b = np.random.randn(4)
preference = 1.0  # A is preferred over B

collector.add_preference(state_a, state_b, preference)
batch_a, batch_b, batch_prefs = collector.sample_batch(32)
print(f"Collected {len(collector)} preferences")`
        }
    ],
    memory: [
        {
            name: 'EpisodicMemory',
            description: 'Stores and retrieves relevant experiences using cosine similarity',
            code: `from rllama import EpisodicMemory, MemoryEntry
import torch

memory = EpisodicMemory(capacity=1000)

entry = MemoryEntry(
    state=torch.randn(64),
    action="take_action",
    reward=1.5,
    importance=0.8
)
memory.add(entry)

query_state = torch.randn(64)
relevant = memory.retrieve_relevant(query_state, k=5)
print(f"Retrieved {len(relevant)} relevant memories")`
        }
    ]
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🦙 DOM loaded, initializing...');
    
    initializeApp();
    setupEventListeners();
    initializeLoadingScreen();
    setupNavigation();
    setupCopyButtons();
    initializeCounters();
    initializeComponentMatrix();
    startTerminalAnimation();
});

function initializeApp() {
    console.log('🦙 RLlama v0.7.0 framework initialized');
    setTimeout(() => {
        document.body.classList.add('loaded');
        loadingComplete = true;
    }, 100);
}

function setupEventListeners() {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', toggleMobileMenu);
    }
    
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize, { passive: true });
}

function initializeLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    const loadingProgress = document.querySelector('.loading-progress');
    
    if (!loadingScreen || !loadingProgress) {
        console.log('�� Loading screen elements not found');
        return;
    }
    
    console.log('🦙 Starting loading animation...');
    
    const steps = [
        { progress: 20, message: 'Loading RewardEngine...' },
        { progress: 40, message: 'Initializing MLPRewardModel...' },
        { progress: 60, message: 'Setting up RLHF pipeline...' },
        { progress: 80, message: 'Loading memory systems...' },
        { progress: 100, message: 'RLlama v0.7.0 ready!' }
    ];
    
    let currentStep = 0;
    
    function updateProgress() {
        if (currentStep >= steps.length) {
            setTimeout(() => {
                hideLoadingScreen();
            }, 300);
            return;
        }
        
        const step = steps[currentStep];
        loadingProgress.style.width = `${step.progress}%`;
        
        const loadingText = document.querySelector('.loading-text span');
        if (loadingText) {
            loadingText.textContent = step.message;
        }
        
        console.log(`🦙 ${step.message} (${step.progress}%)`);
        currentStep++;
        
        setTimeout(updateProgress, 400);
    }
    
    setTimeout(updateProgress, 500);
}

function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        console.log('🦙 Hiding loading screen...');
        loadingScreen.classList.add('fade-out');
        
        setTimeout(() => {
            loadingScreen.style.display = 'none';
            document.body.classList.add('loaded');
            triggerHeroAnimations();
            console.log('🦙 Documentation loaded successfully!');
        }, 500);
    }
}

function triggerHeroAnimations() {
    const heroElements = document.querySelectorAll('.hero-content > *');
    
    heroElements.forEach((element, index) => {
        if (element) {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                element.style.transition = 'opacity 600ms ease, transform 600ms ease';
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, index * 100 + 200);
        }
    });
    
    const terminal = document.querySelector('.terminal-window');
    if (terminal) {
        terminal.style.opacity = '0';
        terminal.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            terminal.style.transition = 'opacity 800ms ease, transform 800ms ease';
            terminal.style.opacity = '1';
            terminal.style.transform = 'scale(1)';
            
            setTimeout(startTerminalAnimation, 500);
        }, 800);
    }
}

function startTerminalAnimation() {
    const codeAnimation = document.getElementById('code-animation');
    if (!codeAnimation) return;
    
    const codeLines = [
        'from rllama import RewardEngine, MLPRewardModel',
        'from rllama.rlhf import PreferenceCollector',
        '',
        '# Initialize composable reward system',
        'engine = RewardEngine("config.yaml", verbose=True)',
        '',
        '# Create neural reward model',
        'model = MLPRewardModel(input_dim=768, hidden_dims=[256, 128])',
        '',
        '# Compute multi-component reward',
        'context = {"response": "Hello, world!", "query": "Greet"}',
        'reward = engine.compute_and_log(context)',
        '',
        'print(f"Reward: {reward:.4f}")',
        '# 🦙 RLlama v0.7.0 - Production Ready!'
    ];
    
    let currentLine = 0;
    let currentChar = 0;
    
    function typeWriter() {
        if (currentLine >= codeLines.length) {
            setTimeout(() => {
                codeAnimation.innerHTML = '';
                currentLine = 0;
                currentChar = 0;
                typeWriter();
            }, 3000);
            return;
        }
        
        const line = codeLines[currentLine];
        
        if (currentChar <= line.length) {
            const displayText = codeLines.slice(0, currentLine).join('\n') + 
                               '\n' + line.substring(0, currentChar) + 
                               (currentChar < line.length ? '█' : '');
            
            codeAnimation.textContent = displayText;
            currentChar++;
            
            setTimeout(typeWriter, 50 + Math.random() * 50);
        } else {
            currentLine++;
            currentChar = 0;
            setTimeout(typeWriter, 500);
        }
    }
    
    typeWriter();
}

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
                
                updateActiveNavItem(link);
                
                if (isMenuOpen) {
                    toggleMobileMenu();
                }
            }
        });
    });
    
    // Make RLlama brand clickable
    const navBrand = document.querySelector('.nav-brand');
    if (navBrand) {
        navBrand.style.cursor = 'pointer';
        navBrand.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
            updateActiveNavItem(document.querySelector('.nav-link[href="#home"]'));
        });
    }
}

function toggleMobileMenu() {
    const navMenu = document.getElementById('nav-menu');
    const navToggle = document.getElementById('nav-toggle');
    
    isMenuOpen = !isMenuOpen;
    
    if (navMenu) navMenu.classList.toggle('active', isMenuOpen);
    if (navToggle) navToggle.classList.toggle('active', isMenuOpen);
    
    document.body.style.overflow = isMenuOpen ? 'hidden' : '';
}

function updateActiveNavItem(activeLink) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

function setupCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            const textToCopy = button.getAttribute('data-copy') || 
                              button.closest('.code-block').querySelector('code').textContent;
            
            if (navigator.clipboard) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showCopySuccess(button);
                });
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = textToCopy;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                showCopySuccess(button);
            }
        });
    });
}

function showCopySuccess(button) {
    const originalIcon = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i>';
    button.style.color = '#00ff66';
    
    setTimeout(() => {
        button.innerHTML = originalIcon;
        button.style.color = '';
    }, 2000);
    
    showToast('Code copied to clipboard! 🦙', 'success');
}

function initializeCounters() {
    const counters = document.querySelectorAll('.stat-number[data-target]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                animateCounter(entry.target);
                entry.target.classList.add('counted');
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-target'));
    const duration = 1500;
    const start = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        const current = Math.floor(progress * target);
        
        element.textContent = current.toString() + (target >= 10 ? '+' : '');
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

function initializeComponentMatrix() {
    const matrixGrid = document.getElementById('component-matrix');
    const matrixButtons = document.querySelectorAll('.matrix-btn');
    
    if (!matrixGrid) return;
    
    matrixButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const category = btn.getAttribute('data-category');
            switchMatrixCategory(category, matrixButtons);
        });
    });
    
    renderComponentMatrix('all');
}

function switchMatrixCategory(category, buttons) {
    buttons.forEach(btn => btn.classList.remove('active'));
    
    const activeBtn = document.querySelector(`[data-category="${category}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    renderComponentMatrix(category);
}

function renderComponentMatrix(category) {
    const matrixGrid = document.getElementById('component-matrix');
    const showcaseDiv = document.getElementById('component-showcase');
    
    if (!matrixGrid) return;
    
    let components = [];
    if (category === 'all') {
        Object.values(componentData).forEach(categoryComponents => {
            components = components.concat(categoryComponents);
        });
    } else {
        components = componentData[category] || [];
    }
    
    matrixGrid.innerHTML = '';
    
    if (components.length === 0) {
        matrixGrid.innerHTML = '<div class="empty-state"><p>No components in this category</p></div>';
        return;
    }
    
    components.forEach((component, index) => {
        const item = document.createElement('div');
        item.className = 'matrix-item';
        item.innerHTML = `
            <h4>${component.name}</h4>
            <p>${component.description}</p>
        `;
        
        item.addEventListener('click', () => {
            showComponentDetails(component, showcaseDiv);
        });
        
        matrixGrid.appendChild(item);
        
        setTimeout(() => {
            item.classList.add('visible');
        }, index * 50);
    });
}

function showComponentDetails(component, showcaseDiv) {
    if (!showcaseDiv) return;
    
    showcaseDiv.innerHTML = `
        <h4>${component.name}</h4>
        <p>${component.description}</p>
        <div class="code-block">
            <div class="code-header">
                <span>Implementation Example</span>
                <button class="copy-btn" data-copy="${component.code.replace(/"/g, '&quot;')}">
                    <i class="fas fa-copy"></i>
                </button>
            </div>
            <pre><code class="language-python">${component.code}</code></pre>
        </div>
    `;
    
    const copyBtn = showcaseDiv.querySelector('.copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const code = copyBtn.getAttribute('data-copy');
            copyToClipboard(code, copyBtn);
        });
    }
}

function copyToClipboard(text, button) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showCopySuccess(button);
        });
    }
}

function handleScroll() {
    scrollPosition = window.pageYOffset;
    updateActiveNavFromScroll();
    updateNavbarBackground();
}

function updateActiveNavFromScroll() {
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
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

function updateNavbarBackground() {
    const navbar = document.getElementById('navbar');
    if (navbar) {
        if (scrollPosition > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }
}

function handleResize() {
    if (window.innerWidth > 768 && isMenuOpen) {
        toggleMobileMenu();
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.textContent = message;
    
    const colors = {
        success: '#00ff66',
        error: '#ff0066',
        info: '#0066ff',
        warning: '#ffcc00'
    };
    
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 24px;
        background: ${colors[type]};
        color: white;
        border-radius: 4px;
        z-index: 10000;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        transform: translateX(100%);
        transition: transform 0.3s ease;
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
        }, 300);
    }, 3000);
}

console.log('🦙 RLlama Documentation Script Loaded Successfully');
