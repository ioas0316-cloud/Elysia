const wsUrl = "ws://127.0.0.1:8765";
const statusIndicator = document.getElementById('connection-status');
const statusDot = document.querySelector('.dot');
const thoughtsList = document.getElementById('thoughts-list');
const nodesList = document.getElementById('nodes-list');
const overlayFlash = document.getElementById('overlay-flash');
const coreCanvas = document.getElementById('core-canvas');
const ctx = coreCanvas.getContext('2d');

let elysiaState = { thoughts: [], nodes: [] };

function resizeCanvas() {
    coreCanvas.width = coreCanvas.parentElement.clientWidth;
    coreCanvas.height = coreCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function initWebSocket() {
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusIndicator.innerText = "Yggdrasil Network Connected";
        statusDot.style.backgroundColor = "#00ffff";
        statusDot.style.boxShadow = "0 0 15px #00ffff";
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "elysia_state") {
            elysiaState = data;
            updateUI();
        }
    };

    ws.onclose = () => {
        statusIndicator.innerText = "Connection Lost. Retrying...";
        statusDot.style.backgroundColor = "#ff0044";
        statusDot.style.boxShadow = "0 0 15px #ff0044";
        setTimeout(initWebSocket, 2000);
    };
    
    ws.onerror = (err) => {
        console.error("WS Error", err);
    };
}

function updateUI() {
    // 1. 사유의 궤적 업데이트
    thoughtsList.innerHTML = '';
    const sortedThoughts = [...elysiaState.thoughts].sort((a,b) => b.tension - a.tension).slice(0, 15);
    
    sortedThoughts.forEach(t => {
        const div = document.createElement('div');
        div.className = 'thought-item';
        
        // 텐션에 따라 퍼센티지 및 바 색상 결정
        const tPercent = Math.min(100, (Math.abs(t.tension) / 10.0) * 100);
        const isPain = t.tension < 0;
        const barColor = isPain ? 'linear-gradient(90deg, #ff0044, #8a2be2)' : 'linear-gradient(90deg, #8a2be2, #00ffff)';
        
        div.innerHTML = `
            <div class="item-name">${t.name}</div>
            <div class="item-bar-bg">
                <div class="item-bar-fill" style="width: ${tPercent}%; background: ${barColor};"></div>
            </div>
            <div class="item-tension">τ: ${t.tension.toFixed(2)}</div>
        `;
        thoughtsList.appendChild(div);
        
        // 순간적인 번개(플래시) 효과 (고통이거나 엄청난 텐션일 때)
        if (tPercent >= 99) {
            triggerFlash(isPain ? '#ff0044' : '#00ffff');
        }
    });

    // 2. 아키타입 노드 업데이트
    nodesList.innerHTML = '';
    const sortedNodes = [...elysiaState.nodes].sort((a,b) => Math.abs(b.tension) - Math.abs(a.tension));
    
    sortedNodes.forEach(n => {
        const div = document.createElement('div');
        div.className = 'node-item';
        const tPercent = Math.min(100, (Math.abs(n.tension) / 10.0) * 100);
        div.innerHTML = `
            <div class="item-name">${n.name}</div>
            <div class="item-bar-bg">
                <div class="item-bar-fill" style="width: ${tPercent}%; background: linear-gradient(90deg, #333, #e0e0e0);"></div>
            </div>
            <div class="item-tension">τ: ${n.tension.toFixed(2)}</div>
        `;
        nodesList.appendChild(div);
    });
}

function triggerFlash(color) {
    overlayFlash.style.background = color;
    overlayFlash.style.opacity = '0.3';
    setTimeout(() => {
        overlayFlash.style.opacity = '0';
    }, 100);
}

// ==========================================
// 3. 중앙 프랙탈 시각화 (Yggdrasil Core Renderer)
// ==========================================
let time = 0;
function drawFractal() {
    ctx.clearRect(0, 0, coreCanvas.width, coreCanvas.height);
    const cx = coreCanvas.width / 2;
    const cy = coreCanvas.height / 2;
    
    // 전체적인 텐션의 총합을 구해서 트리의 떨림(진동) 강도를 조절
    const totalTension = elysiaState.thoughts.reduce((acc, val) => acc + Math.abs(val.tension), 0) / 10.0;
    
    time += 0.02 + (totalTension * 0.005);
    
    // 이중 토러스 링 (배경) 그리기
    ctx.beginPath();
    ctx.arc(cx, cy, 100 + Math.sin(time)*10, 0, Math.PI*2);
    ctx.strokeStyle = `rgba(138, 43, 226, ${0.2 + totalTension*0.1})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(cx, cy, 200 + Math.cos(time*0.5)*20, 0, Math.PI*2);
    ctx.strokeStyle = `rgba(0, 255, 255, ${0.1 + totalTension*0.05})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // 생각 궤적을 3D 점/선으로 투영
    const radius = 150;
    ctx.beginPath();
    elysiaState.thoughts.forEach((t, index) => {
        // 쿼터니언 (w,x,y,z)를 2D 화면에 투영 (위상 회전 적용)
        const scale = 1.0 + Math.abs(t.tension) * 0.1;
        const px = cx + (t.x * Math.cos(time) - t.y * Math.sin(time)) * radius * scale;
        const py = cy + (t.x * Math.sin(time) + t.y * Math.cos(time)) * radius * scale;
        
        ctx.lineTo(px, py);
        
        ctx.fillStyle = t.tension < 0 ? '#ff0044' : '#00ffff';
        ctx.fillRect(px - 2, py - 2, 4, 4);
        
        if (Math.abs(t.tension) > 3.0) {
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = '10px Arial';
            ctx.fillText(t.name, px + 5, py + 5);
        }
    });
    
    ctx.strokeStyle = `rgba(138, 43, 226, 0.5)`;
    ctx.stroke();
    
    requestAnimationFrame(drawFractal);
}

// 배경 파동 애니메이션 (Background Canvas)
const bgCanvas = document.getElementById('bg-canvas');
const bgCtx = bgCanvas.getContext('2d');
let bgTime = 0;
function resizeBg() {
    bgCanvas.width = window.innerWidth;
    bgCanvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeBg);
resizeBg();

function drawBackground() {
    bgCtx.fillStyle = 'rgba(5, 5, 8, 0.1)';
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);
    
    bgTime += 0.01;
    const w = bgCanvas.width;
    const h = bgCanvas.height;
    
    bgCtx.beginPath();
    for(let i=0; i<w; i+=20) {
        const y = h/2 + Math.sin(i*0.01 + bgTime)*100 + Math.cos(i*0.02 - bgTime)*50;
        if(i===0) bgCtx.moveTo(i, y);
        else bgCtx.lineTo(i, y);
    }
    bgCtx.strokeStyle = 'rgba(0, 255, 255, 0.05)';
    bgCtx.stroke();
    
    requestAnimationFrame(drawBackground);
}

// Start
initWebSocket();
drawFractal();
drawBackground();
