// Custom JavaScript for AI Fitness Tracker Documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to anchor links
    addSmoothScrolling();
    
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Add badges to sections
    addSectionBadges();
    
    // Enhance tables with sorting
    enhanceTables();
    
    // Add back to top button
    addBackToTop();
    
    // Add keyboard navigation
    addKeyboardNavigation();
});

function addSmoothScrolling() {
    // Add smooth scrolling to all anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function addCopyButtons() {
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = '📋 Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: #007acc;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        wrapper.appendChild(button);
        
        wrapper.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
        });
        
        wrapper.addEventListener('mouseleave', () => {
            button.style.opacity = '0';
        });
        
        button.addEventListener('click', () => {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.innerHTML = '✅ Copied!';
                setTimeout(() => {
                    button.innerHTML = '📋 Copy';
                }, 2000);
            });
        });
    });
}

function addSectionBadges() {
    // Add badges to different types of content
    const headings = document.querySelectorAll('h1, h2, h3');
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        let badge = null;
        
        if (text.includes('api') || text.includes('reference')) {
            badge = createBadge('API', 'badge-api');
        } else if (text.includes('tutorial') || text.includes('guide')) {
            badge = createBadge('Tutorial', 'badge-tutorial');
        } else if (text.includes('example') || text.includes('demo')) {
            badge = createBadge('Example', 'badge-example');
        } else if (text.includes('advanced') || text.includes('development')) {
            badge = createBadge('Advanced', 'badge-advanced');
        }
        
        if (badge) {
            heading.insertBefore(badge, heading.firstChild);
        }
    });
}

function createBadge(text, className) {
    const badge = document.createElement('span');
    badge.className = `badge ${className}`;
    badge.textContent = text;
    return badge;
}

function enhanceTables() {
    // Add sorting functionality to tables
    const tables = document.querySelectorAll('table.docutils');
    tables.forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.style.position = 'relative';
            header.title = 'Click to sort';
            
            const sortIcon = document.createElement('span');
            sortIcon.innerHTML = ' ↕️';
            sortIcon.style.fontSize = '0.8em';
            header.appendChild(sortIcon);
            
            header.addEventListener('click', () => {
                sortTable(table, index);
            });
        });
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const isNumeric = rows.every(row => {
        const cell = row.cells[columnIndex];
        return cell && !isNaN(parseFloat(cell.textContent));
    });
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex]?.textContent || '';
        const bText = b.cells[columnIndex]?.textContent || '';
        
        if (isNumeric) {
            return parseFloat(aText) - parseFloat(bText);
        } else {
            return aText.localeCompare(bText);
        }
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

function addBackToTop() {
    // Create back to top button
    const button = document.createElement('button');
    button.innerHTML = '↑';
    button.className = 'back-to-top';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: #2980B9;
        color: white;
        border: none;
        font-size: 20px;
        cursor: pointer;
        display: none;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    `;
    
    document.body.appendChild(button);
    
    // Show/hide button based on scroll position
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            button.style.display = 'block';
        } else {
            button.style.display = 'none';
        }
    });
    
    // Smooth scroll to top
    button.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

function addKeyboardNavigation() {
    // Add keyboard shortcuts for navigation
    document.addEventListener('keydown', (e) => {
        // Ctrl + / to focus search
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            const searchInput = document.querySelector('input[type="text"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Arrow keys for navigation
        if (e.altKey) {
            const currentUrl = window.location.href;
            const navLinks = document.querySelectorAll('.wy-menu-vertical a');
            const currentIndex = Array.from(navLinks).findIndex(link => 
                link.href === currentUrl
            );
            
            if (e.key === 'ArrowLeft' && currentIndex > 0) {
                navLinks[currentIndex - 1].click();
            } else if (e.key === 'ArrowRight' && currentIndex < navLinks.length - 1) {
                navLinks[currentIndex + 1].click();
            }
        }
    });
}

// Add loading animation for images
function addImageLoadingEffects() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.style.transition = 'opacity 0.3s ease';
        img.style.opacity = '0';
        
        img.addEventListener('load', () => {
            img.style.opacity = '1';
        });
        
        // If image is already loaded (cached)
        if (img.complete) {
            img.style.opacity = '1';
        }
    });
}

// Add syntax highlighting enhancements
function enhanceSyntaxHighlighting() {
    // Add line numbers to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const lines = block.textContent.split('\n');
        if (lines.length > 3) { // Only add line numbers for longer code blocks
            const lineNumbers = lines.map((_, index) => 
                `<span class="line-number">${index + 1}</span>`
            ).join('\n');
            
            const lineNumberDiv = document.createElement('div');
            lineNumberDiv.className = 'line-numbers';
            lineNumberDiv.innerHTML = lineNumbers;
            lineNumberDiv.style.cssText = `
                position: absolute;
                left: 0;
                top: 0;
                padding: 1.5rem 0.5rem;
                background: #f0f0f0;
                border-right: 1px solid #ddd;
                font-family: monospace;
                font-size: 0.9em;
                color: #999;
                user-select: none;
                line-height: 1.5;
            `;
            
            block.parentElement.style.position = 'relative';
            block.parentElement.style.paddingLeft = '3rem';
            block.parentElement.appendChild(lineNumberDiv);
        }
    });
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    addImageLoadingEffects();
    enhanceSyntaxHighlighting();
    
    // Add search enhancement
    const searchInput = document.querySelector('input[type="text"]');
    if (searchInput) {
        searchInput.placeholder = '🔍 Search documentation...';
        searchInput.addEventListener('focus', function() {
            this.style.boxShadow = '0 0 10px rgba(41, 128, 185, 0.5)';
        });
        
        searchInput.addEventListener('blur', function() {
            this.style.boxShadow = 'none';
        });
    }
});
