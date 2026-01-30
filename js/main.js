document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Mobile Menu Toggle
    const mobileToggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.nav-links');
    const navItems = document.querySelectorAll('.nav-links a');

    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }

    // Close menu when a link is clicked
    navItems.forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
        });
    });

    // 2. Load Projects from JSON
    const projectsContainer = document.getElementById('projects-container');
    if (projectsContainer) {
        fetch('projects.json')
            .then(response => response.json())
            .then(data => {
                let html = '';
                data.forEach((project, index) => {
                    html += `
                        <article class="project-card reveal">
                            <a href="${project.link}" target="_blank" class="project-img" style="background-image: url('${project.image}');" aria-label="${project.title}">
                                <div class="project-overlay">
                                    <h3>${project.title}</h3>
                                    <span>${project.subtitle}</span>
                                </div>
                            </a>
                            <div class="project-content">
                                <h3>${project.title}</h3>
                                <h4>Product Vision</h4>
                                <ul><li>${project.vision}</li></ul>
                                <h4>Problem</h4>
                                <ul><li>${project.problem}</li></ul>
                                <h4>Solution</h4>
                                <ul><li>${project.solution}</li></ul>
                                <h4>Impact</h4>
                                <ul><li>${project.impact}</li></ul>
                            </div>
                        </article>
                    `;
                });
                projectsContainer.innerHTML = html;
                
                // Re-trigger observer for new elements
                observeElements();
            })
            .catch(error => console.error('Error loading projects:', error));
    }

    // 3. Scroll Animations (Intersection Observer)
    function observeElements() {
        const observerOptions = {
            threshold: 0.15, // Trigger when 15% of element is visible
            rootMargin: "0px 0px -50px 0px"
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('active');
                    observer.unobserve(entry.target); // Only animate once
                }
            });
        }, observerOptions);

        document.querySelectorAll('.reveal').forEach(el => {
            observer.observe(el);
        });
    }

    // Initial observation
    observeElements();
});