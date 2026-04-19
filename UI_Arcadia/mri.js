// SSE Connection
const eventSource = new EventSource('http://localhost:8080/stream');
const dot = document.getElementById('connection-dot');
const statusText = document.getElementById('connection-text');

// DOM Elements
const valNodes = document.getElementById('val-active-nodes');
const valEdges = document.getElementById('val-edges');
const barJoy = document.getElementById('bar-joy');
const barCoherence = document.getElementById('bar-coherence');
const barEnthalpy = document.getElementById('bar-enthalpy');
const alertPanel = document.getElementById('bottleneck-alert');
const logContainer = document.getElementById('log-container');

// Chart Setup
const ctx = document.getElementById('activeNodesChart').getContext('2d');
const maxDataPoints = 100;
const chartData = {
    labels: Array(maxDataPoints).fill(''),
    datasets: [{
        label: 'Active Cells',
        data: Array(maxDataPoints).fill(0),
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0
    }]
};

const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
        y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#8b9bb4' },
            title: { display: true, text: 'Cell Count', color: '#8b9bb4' }
        },
        x: {
            grid: { display: false },
            ticks: { display: false }
        }
    },
    plugins: {
        legend: { display: false }
    }
};

const mriChart = new Chart(ctx, {
    type: 'line',
    data: chartData,
    options: chartOptions
});

function addLog(msg, isSpike = false) {
    const p = document.createElement('p');
    p.className = `log-entry ${isSpike ? 'spike' : ''}`;
    const time = new Date().toLocaleTimeString();
    p.innerText = `[${time}] ${msg}`;
    logContainer.prepend(p);
    if (logContainer.children.length > 20) {
        logContainer.lastChild.remove();
    }
}

let lastNodes = 0;
const BOTTLENECK_THRESHOLD = 500000; // 500k active nodes warning

eventSource.onopen = () => {
    dot.classList.add('connected');
    statusText.innerText = 'Receiving Somatic Telemetry...';
    addLog('Telemetry connection established.');
};

eventSource.onerror = () => {
    dot.classList.remove('connected');
    statusText.innerText = 'Connection lost';
};

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update Metrics
    const active = data.active_nodes || 0;
    valNodes.innerText = active.toLocaleString();
    valEdges.innerText = (data.edges || 0).toLocaleString();
    
    barJoy.style.width = `${Math.min(100, (data.joy || 0) * 100)}%`;
    barCoherence.style.width = `${Math.min(100, (data.coherence || 0) * 100)}%`;
    barEnthalpy.style.width = `${Math.min(100, (data.enthalpy || 0) * 100)}%`;

    // Alert Logic
    if (active > BOTTLENECK_THRESHOLD) {
        alertPanel.classList.add('active');
        if (active > lastNodes + 100000) {
            addLog(`Tectonic Spike Detected: ${active.toLocaleString()} cells active!`, true);
        }
    } else {
        alertPanel.classList.remove('active');
    }
    lastNodes = active;

    // Update Chart
    chartData.datasets[0].data.push(active);
    chartData.datasets[0].data.shift();
    
    // Adjust max Y scale dynamically if spike is huge
    if (active > mriChart.options.scales.y.max || !mriChart.options.scales.y.max) {
        mriChart.options.scales.y.max = Math.max(1000, active * 1.2);
    } else if (active < mriChart.options.scales.y.max * 0.1 && mriChart.options.scales.y.max > 1000) {
         // Slowly decay the max scale if activity drops
        mriChart.options.scales.y.max = mriChart.options.scales.y.max * 0.95;
    }

    mriChart.update();
};
