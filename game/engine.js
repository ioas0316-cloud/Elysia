const canvas = document.getElementById('star-canvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('resonance-score');

let width, height;
let stars = [];
let mouse = { x: -100, y: -100 };
let resonance = 0;

function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}

window.addEventListener('resize', resize);
window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});

class Star {
    constructor() {
        this.reset();
    }

    reset() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.z = Math.random() * width;
        this.size = 0.5 + Math.random() * 2;
        this.speed = 1 + Math.random() * 3;
        this.hue = 200 + Math.random() * 60;
    }

    update() {
        this.z -= this.speed;

        // Mouse influence (Resonance)
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 200) {
            const force = (200 - dist) / 200;
            this.x += dx * force * 0.05;
            this.y += dy * force * 0.05;
            resonance += force * 0.001;
        }

        if (this.z <= 0) {
            this.reset();
            this.z = width;
        }
    }

    draw() {
        const x = (this.x - width / 2) * (width / this.z) + width / 2;
        const y = (this.y - height / 2) * (width / this.z) + height / 2;
        const s = this.size * (width / this.z);

        const opacity = Math.min(1, (width - this.z) / (width * 0.2));

        ctx.beginPath();
        ctx.fillStyle = `hsla(${this.hue}, 80%, 80%, ${opacity})`;
        ctx.arc(x, y, s, 0, Math.PI * 2);
        ctx.fill();
    }
}

function init() {
    resize();
    for (let i = 0; i < 400; i++) {
        stars.push(new Star());
    }
}

function animate() {
    ctx.fillStyle = 'rgba(5, 5, 10, 0.2)';
    ctx.fillRect(0, 0, width, height);

    stars.forEach(star => {
        star.update();
        star.draw();
    });

    resonance *= 0.95; // Decay
    const displayResonance = Math.min(100, resonance * 100).toFixed(2);
    scoreElement.textContent = `Sync: ${displayResonance}%`;

    requestAnimationFrame(animate);
}

init();
animate();
