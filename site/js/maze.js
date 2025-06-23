document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('maze-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    let size, cellSize, maze, ball, path;

    // A predefined path for the ball to follow (indices of the maze grid)
    const pathCoords = [
        { x: 1, y: 1 }, { x: 2, y: 1 }, { x: 3, y: 1 }, { x: 4, y: 1 }, { x: 5, y: 1 },
        { x: 5, y: 2 }, { x: 5, y: 3 }, { x: 4, y: 3 }, { x: 3, y: 3 }, { x: 3, y: 4 },
        { x: 3, y: 5 }, { x: 3, y: 6 }, { x: 3, y: 7 }, { x: 4, y: 7 }, { x: 5, y: 7 },
        { x: 6, y: 7 }, { x: 7, y: 7 }, { x: 7, y: 6 }, { x: 7, y: 5 }, { x: 7, y: 4 },
        { x: 7, y: 3 }, { x: 7, y: 2 }, { x: 7, y: 1 }, { x: 8, y: 1 }, { x: 9, y: 1 }
    ];
    
    let pathIndex = 0;
    let progress = 0;
    const animationSpeed = 0.03;

    const mazeLayout = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    
    function init() {
        resizeCanvas();
        maze = mazeLayout;
        path = pathCoords;
        ball = {
            radius: cellSize / 3,
            color: '#ffd700' // Yellow
        };
        updateBallPosition();
        animate();
    }

    function resizeCanvas() {
        size = Math.min(container.offsetWidth, container.offsetHeight);
        canvas.width = size;
        canvas.height = size;
        cellSize = size / maze[0].length;
        if (ball) ball.radius = cellSize / 3;
    }

    function drawMaze() {
        const wallColor = getComputedStyle(document.body).getPropertyValue('--maze-wall-color').trim();
        ctx.fillStyle = wallColor;
        for (let y = 0; y < maze.length; y++) {
            for (let x = 0; x < maze[y].length; x++) {
                if (maze[y][x] === 1) {
                    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                }
            }
        }
    }

    function drawBall() {
        ctx.beginPath();
        ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
        ctx.fillStyle = ball.color;
        ctx.fill();
        ctx.closePath();
    }
    
    function updateBallPosition() {
        const currentPoint = path[pathIndex];
        const nextPoint = path[(pathIndex + 1) % path.length];

        const targetX = (currentPoint.x + (nextPoint.x - currentPoint.x) * progress) * cellSize + cellSize / 2;
        const targetY = (currentPoint.y + (nextPoint.y - currentPoint.y) * progress) * cellSize + cellSize / 2;
        
        ball.x = targetX;
        ball.y = targetY;

        progress += animationSpeed;
        if (progress >= 1) {
            progress = 0;
            pathIndex = (pathIndex + 1) % path.length;
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawMaze();
        updateBallPosition();
        drawBall();
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', init);
    
    // Use a MutationObserver to detect theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === "class") {
                // Redraw maze for new theme color
                drawMaze();
            }
        });
    });
    observer.observe(document.body, { attributes: true });

    init();
});