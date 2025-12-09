/**
 * Precision Lab Visualizations
 *
 * Interactive Chart.js visualizations for precision-performance tradeoffs
 * in the power method algorithm.
 */

// Precision format colors
const COLORS = {
    fp8: '#ff6b35',
    fp16: '#4a90e2',
    fp32: '#50c878',
    fp64: '#9b59b6',
    background: '#0a0e27',
    grid: '#283593',
    text: '#e8eaf6',
    textSecondary: '#9fa8da'
};

/**
 * Base class for animated visualizations
 */
class AnimatedVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.animationId = null;
        this.isPlaying = false;
        this.speed = 2;
        this.currentFrame = 0;
    }

    play() {
        if (!this.isPlaying) {
            this.isPlaying = true;
            this.animate();
        }
    }

    pause() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    reset() {
        this.pause();
        this.currentFrame = 0;
        this.updateVisualization();
    }

    setSpeed(speed) {
        this.speed = speed;
    }

    animate() {
        if (!this.isPlaying) return;

        this.currentFrame += this.speed;
        this.updateVisualization();

        if (this.currentFrame < this.maxFrames) {
            this.animationId = requestAnimationFrame(() => this.animate());
        } else {
            this.isPlaying = false;
        }
    }

    updateVisualization() {
        // Override in subclasses
    }
}

/**
 * Precision Race Visualization
 * Animated convergence curves for FP8, FP16, FP32, FP64
 */
class PrecisionRaceVisualization extends AnimatedVisualization {
    constructor(canvasId, traceData) {
        super(canvasId);

        this.traces = {
            fp8: traceData.fp8,
            fp16: traceData.fp16,
            fp32: traceData.fp32,
            fp64: traceData.fp64
        };

        // Find max iterations across all traces
        this.maxFrames = Math.max(
            this.traces.fp8.length,
            this.traces.fp16.length,
            this.traces.fp32.length,
            this.traces.fp64.length
        );

        this.initChart();
    }

    initChart() {
        const ctx = this.ctx;

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'FP8 (E4M3)',
                        data: [],
                        borderColor: COLORS.fp8,
                        backgroundColor: COLORS.fp8 + '20',
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    {
                        label: 'FP16',
                        data: [],
                        borderColor: COLORS.fp16,
                        backgroundColor: COLORS.fp16 + '20',
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    {
                        label: 'FP32',
                        data: [],
                        borderColor: COLORS.fp32,
                        backgroundColor: COLORS.fp32 + '20',
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    {
                        label: 'FP64',
                        data: [],
                        borderColor: COLORS.fp64,
                        backgroundColor: COLORS.fp64 + '20',
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Iteration',
                            color: COLORS.text,
                            font: { size: 14 }
                        },
                        grid: {
                            color: COLORS.grid
                        },
                        ticks: {
                            color: COLORS.textSecondary
                        }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Residual Norm',
                            color: COLORS.text,
                            font: { size: 14 }
                        },
                        grid: {
                            color: COLORS.grid
                        },
                        ticks: {
                            color: COLORS.textSecondary,
                            callback: function(value) {
                                return value.toExponential(0);
                            }
                        },
                        min: 1e-16,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: COLORS.background,
                        titleColor: COLORS.text,
                        bodyColor: COLORS.textSecondary,
                        borderColor: COLORS.grid,
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                return `${label}: ${value.toExponential(2)}`;
                            }
                        }
                    }
                }
            }
        });

        this.updateVisualization();
    }

    updateVisualization() {
        const frame = Math.floor(this.currentFrame);

        // Update each dataset with data up to current frame
        const datasets = [
            { trace: this.traces.fp8, index: 0 },
            { trace: this.traces.fp16, index: 1 },
            { trace: this.traces.fp32, index: 2 },
            { trace: this.traces.fp64, index: 3 }
        ];

        datasets.forEach(({ trace, index }) => {
            const data = [];
            const maxIdx = Math.min(frame, trace.length - 1);

            for (let i = 0; i <= maxIdx; i++) {
                data.push({
                    x: trace[i].iteration,
                    y: trace[i].residual_norm
                });
            }

            this.chart.data.datasets[index].data = data;

            // Update iteration counter
            const precisionName = ['fp8', 'fp16', 'fp32', 'fp64'][index];
            const elem = document.getElementById(`${precisionName}-iters`);
            if (elem) {
                elem.textContent = maxIdx;
            }
        });

        this.chart.update();
    }
}

/**
 * Cascading Precision Visualization
 * Shows FP8→FP16→FP32→FP64 transitions over time
 */
class CascadingPrecisionVisualization extends AnimatedVisualization {
    constructor(canvasId, cascadeData) {
        super(canvasId);

        this.metadata = cascadeData.metadata;
        this.segments = cascadeData.segments;
        this.trace = cascadeData.trace;
        this.maxFrames = this.trace.length;

        // Map precision names to colors
        this.precisionColors = {
            'FP8': COLORS.fp8,
            'FP16': COLORS.fp16,
            'FP32': COLORS.fp32,
            'FP64': COLORS.fp64
        };

        this.initChart();
    }

    initChart() {
        const ctx = this.ctx;

        // Create one dataset per segment
        const datasets = this.segments.map((seg, idx) => ({
            label: seg.precision,
            data: [],
            borderColor: this.precisionColors[seg.precision],
            backgroundColor: this.precisionColors[seg.precision] + '30',
            borderWidth: 4,
            pointRadius: 0,
            segment: {
                borderColor: ctx => {
                    // Show different color after segment ends
                    return this.precisionColors[seg.precision];
                }
            },
            tension: 0.1
        }));

        this.chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Iteration',
                            color: COLORS.text,
                            font: { size: 14 }
                        },
                        grid: {
                            color: COLORS.grid
                        },
                        ticks: {
                            color: COLORS.textSecondary
                        }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Residual Norm',
                            color: COLORS.text,
                            font: { size: 14 }
                        },
                        grid: {
                            color: COLORS.grid
                        },
                        ticks: {
                            color: COLORS.textSecondary,
                            callback: function(value) {
                                return value.toExponential(0);
                            }
                        },
                        min: 1e-16,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: COLORS.background,
                        titleColor: COLORS.text,
                        bodyColor: COLORS.textSecondary,
                        borderColor: COLORS.grid,
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                return `${label}: ${value.toExponential(2)}`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {}
                    }
                }
            }
        });

        this.updateVisualization();
    }

    updateVisualization() {
        const frame = Math.floor(this.currentFrame);
        const maxIdx = Math.min(frame, this.trace.length - 1);

        // Determine current segment and precision
        let currentSegment = this.segments[0];
        for (const seg of this.segments) {
            if (maxIdx >= seg.start_iteration && maxIdx <= seg.end_iteration) {
                currentSegment = seg;
                break;
            }
        }

        // Update precision indicator
        const precisionElem = document.getElementById('current-precision');
        const transitionElem = document.getElementById('transition-info');
        if (precisionElem) {
            precisionElem.textContent = currentSegment.precision;
            precisionElem.style.color = this.precisionColors[currentSegment.precision];
        }
        if (transitionElem) {
            const iterCount = maxIdx - currentSegment.start_iteration;
            transitionElem.textContent = `Iteration ${iterCount} in ${currentSegment.precision} segment`;
        }

        // Update datasets
        this.segments.forEach((seg, idx) => {
            const data = [];
            const segStart = seg.start_iteration;
            const segEnd = Math.min(seg.end_iteration, maxIdx);

            if (maxIdx >= segStart) {
                for (let i = segStart; i <= segEnd; i++) {
                    const point = this.trace[i];
                    if (point) {
                        data.push({
                            x: point.iteration,
                            y: point.residual_norm
                        });
                    }
                }
            }

            this.chart.data.datasets[idx].data = data;
        });

        // Update iteration counters
        const counters = { FP8: 0, FP16: 0, FP32: 0, FP64: 0 };
        for (const seg of this.segments) {
            if (maxIdx >= seg.end_iteration) {
                counters[seg.precision] = seg.iterations;
            } else if (maxIdx >= seg.start_iteration) {
                counters[seg.precision] = maxIdx - seg.start_iteration;
            }
        }

        document.getElementById('fp8-iters').textContent = counters.FP8;
        document.getElementById('fp16-iters').textContent = counters.FP16;
        document.getElementById('fp32-iters').textContent = counters.FP32;
        document.getElementById('fp64-iters').textContent = counters.FP64;

        this.chart.update();
    }
}

// Export for use in HTML pages
window.PrecisionRaceVisualization = PrecisionRaceVisualization;
window.CascadingPrecisionVisualization = CascadingPrecisionVisualization;
