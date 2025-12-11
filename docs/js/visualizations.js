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

        // Handle both old (array) and new (object with trace) formats
        this.traces = {
            fp8: traceData.fp8.trace || traceData.fp8,
            fp16: traceData.fp16.trace || traceData.fp16,
            fp32: traceData.fp32.trace || traceData.fp32,
            fp64: traceData.fp64.trace || traceData.fp64
        };

        // Store metadata if available
        this.metadata = {
            fp8: traceData.fp8.metadata || null,
            fp16: traceData.fp16.metadata || null,
            fp32: traceData.fp32.metadata || null,
            fp64: traceData.fp64.metadata || null
        };

        // Speedup factors: how many iterations of each precision = 1 FP64 iteration
        // Lower precision has higher throughput, so divide iterations by speedup factor
        this.speedupFactors = {
            fp8: 6,   // FP8 is 6× faster, so 1 FP8 iter = 1/6 effective FP64 iter
            fp16: 4,  // FP16 is 4× faster
            fp32: 2,  // FP32 is 2× faster
            fp64: 1   // FP64 is baseline (1:1)
        };

        // Calculate max effective iterations for animation
        // Each precision's effective iterations = raw iterations / speedup
        const maxEffectiveIterations = Math.max(
            this.traces.fp8.length / this.speedupFactors.fp8,
            this.traces.fp16.length / this.speedupFactors.fp16,
            this.traces.fp32.length / this.speedupFactors.fp32,
            this.traces.fp64.length / this.speedupFactors.fp64
        );
        this.maxFrames = Math.ceil(maxEffectiveIterations);

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
                        borderWidth: 4,
                        pointRadius: 0,
                        tension: 0.1,
                        borderDash: [10, 5]  // Dashed line for FP8
                    },
                    {
                        label: 'FP16',
                        data: [],
                        borderColor: COLORS.fp16,
                        backgroundColor: COLORS.fp16 + '20',
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1,
                        borderDash: [5, 3]  // Dotted line for FP16
                    },
                    {
                        label: 'FP32',
                        data: [],
                        borderColor: COLORS.fp32,
                        backgroundColor: COLORS.fp32 + '20',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        tension: 0.1
                        // Solid line for FP32
                    },
                    {
                        label: 'FP64',
                        data: [],
                        borderColor: COLORS.fp64,
                        backgroundColor: COLORS.fp64 + '20',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                        // Solid line for FP64
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'nearest'  // Use nearest since x values differ per dataset due to speedup scaling
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Effective FP64 Iterations (normalized by throughput)',
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
                        display: true,
                        labels: {
                            color: COLORS.text,
                            usePointStyle: true,
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        backgroundColor: COLORS.background,
                        titleColor: COLORS.text,
                        bodyColor: COLORS.textSecondary,
                        borderColor: COLORS.grid,
                        borderWidth: 1,
                        callbacks: {
                            title: function(tooltipItems) {
                                if (tooltipItems.length === 0) return '';
                                const effectiveIter = tooltipItems[0].parsed.x;
                                return `Effective iteration: ${effectiveIter.toFixed(1)}`;
                            },
                            label: (context) => {
                                const label = context.dataset.label || '';
                                const residual = context.parsed.y;
                                const effectiveIter = context.parsed.x;
                                // Calculate raw iteration from effective iteration using speedup factor
                                const speedups = { 'FP8 (E4M3)': 6, 'FP16': 4, 'FP32': 2, 'FP64': 1 };
                                const speedup = speedups[label] || 1;
                                const rawIter = Math.round(effectiveIter * speedup);
                                return `${label}: ${residual.toExponential(2)} (raw iter: ${rawIter})`;
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            modifierKey: 'shift'  // Hold shift to pan
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                                modifierKey: 'ctrl'  // Ctrl+wheel to zoom
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'xy',
                            onZoomComplete: function({chart}) {
                                // Update axis labels to show zoom state
                                chart.update('none');
                            }
                        },
                        limits: {
                            x: { min: 0, max: 600 },  // Max effective iterations (3000/6 = 500) + buffer
                            y: { min: 1e-16, max: 1 }
                        }
                    }
                }
            }
        });

        // Store reference for reset
        this.chartInstance = this.chart;
        this.updateVisualization();
    }

    resetZoom() {
        if (this.chart) {
            this.chart.resetZoom();
        }
    }

    updateVisualization() {
        // Current frame represents effective FP64 iterations
        const effectiveFrame = this.currentFrame;

        // Update each dataset with data up to current effective frame
        // X-axis is scaled by throughput: FP8/6, FP16/4, FP32/2, FP64/1
        const datasets = [
            { trace: this.traces.fp8, index: 0, name: 'fp8', speedup: this.speedupFactors.fp8 },
            { trace: this.traces.fp16, index: 1, name: 'fp16', speedup: this.speedupFactors.fp16 },
            { trace: this.traces.fp32, index: 2, name: 'fp32', speedup: this.speedupFactors.fp32 },
            { trace: this.traces.fp64, index: 3, name: 'fp64', speedup: this.speedupFactors.fp64 }
        ];

        datasets.forEach(({ trace, index, name, speedup }) => {
            const data = [];

            // Calculate how many raw iterations correspond to current effective frame
            // effectiveFrame = rawIteration / speedup, so rawIteration = effectiveFrame * speedup
            const maxRawIteration = Math.floor(effectiveFrame * speedup);
            const traceEnd = Math.min(maxRawIteration, trace.length - 1);

            // Add trace data up to current raw iteration, with x scaled to effective iterations
            for (let i = 0; i <= traceEnd; i++) {
                data.push({
                    x: trace[i].iteration / speedup,  // Convert to effective FP64 iterations
                    y: trace[i].residual_norm
                });
            }

            this.chart.data.datasets[index].data = data;

            // Update iteration counter (show raw iterations)
            const iterElem = document.getElementById(`${name}-iters`);
            if (iterElem) {
                iterElem.textContent = `${traceEnd} iters`;
            }

            // Update residual display
            const residualElem = document.getElementById(`${name}-residual`);
            if (residualElem && traceEnd >= 0 && trace[traceEnd]) {
                residualElem.textContent = trace[traceEnd].residual_norm.toExponential(2);
            }
        });

        this.chart.update();
    }
}

/**
 * Cascading Precision Visualization
 * Shows FP8→FP16→FP32→FP64 transitions vs FP64 baseline
 * X-axis is "Effective FP64 Iterations" to show real performance advantage
 */
class CascadingPrecisionVisualization extends AnimatedVisualization {
    constructor(canvasId, cascadeData, fp64Data = null) {
        super(canvasId);

        this.metadata = cascadeData.metadata;
        this.segments = cascadeData.segments;
        this.trace = cascadeData.trace;

        // FP64 reference trace (handle both old array and new object formats)
        this.fp64Trace = fp64Data ? (fp64Data.trace || fp64Data) : null;

        // Speedup factors: how many iterations of each precision = 1 FP64 iteration
        this.speedupFactors = {
            'FP8': 6,   // FP8 is 6× faster, so divide by 6
            'FP16': 4,  // FP16 is 4× faster
            'FP32': 2,  // FP32 is 2× faster
            'FP64': 1   // FP64 is baseline
        };

        // Calculate max frames for animation
        // Animation runs until cascading completes (FP64 is limited to same budget)
        this.maxFrames = this.trace.length + 20; // +20 buffer for smooth ending

        // Map precision names to colors
        this.precisionColors = {
            'FP8': COLORS.fp8,
            'FP16': COLORS.fp16,
            'FP32': COLORS.fp32,
            'FP64': COLORS.fp64
        };

        this.initChart();
    }

    // Convert raw iteration to effective FP64 iteration
    toEffectiveIteration(rawIteration, precision) {
        return rawIteration / this.speedupFactors[precision];
    }

    // Get cumulative effective iteration for a point in the cascading trace
    getCumulativeEffective(iterationIndex) {
        let effective = 0;
        for (const seg of this.segments) {
            if (iterationIndex >= seg.end_iteration) {
                // Full segment completed
                effective += seg.iterations / this.speedupFactors[seg.precision];
            } else if (iterationIndex >= seg.start_iteration) {
                // Partial segment
                const iterationsInSeg = iterationIndex - seg.start_iteration;
                effective += iterationsInSeg / this.speedupFactors[seg.precision];
                break;
            }
        }
        return effective;
    }

    initChart() {
        const ctx = this.ctx;

        // Create datasets: one per cascading segment + FP64 reference
        const datasets = this.segments.map((seg, idx) => ({
            label: `Cascading (${seg.precision})`,
            data: [],
            borderColor: this.precisionColors[seg.precision],
            backgroundColor: this.precisionColors[seg.precision] + '30',
            borderWidth: 4,
            pointRadius: 0,
            tension: 0.1
        }));

        // Add FP64 reference line (dashed)
        if (this.fp64Trace) {
            datasets.push({
                label: 'FP64 Only (Reference)',
                data: [],
                borderColor: COLORS.fp64,
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [8, 4],
                pointRadius: 0,
                tension: 0.1
            });
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'nearest'  // Use nearest point, not array index
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Effective FP64 Iterations (normalized by speedup)',
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
                        display: true,
                        position: 'top',
                        labels: {
                            color: COLORS.text,
                            usePointStyle: true
                        }
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
                                const x = context.parsed.x.toFixed(1);
                                return `${label}: ${value.toExponential(2)} @ ${x} eff. iters`;
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            modifierKey: 'shift'  // Hold shift to pan
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                                modifierKey: 'ctrl'  // Ctrl+wheel to zoom
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'xy'
                        },
                        limits: {
                            x: { min: 0, max: 200 },  // ~165 effective iterations + buffer
                            y: { min: 1e-16, max: 1 }
                        }
                    }
                }
            }
        });

        this.updateVisualization();
    }

    resetZoom() {
        if (this.chart) {
            this.chart.resetZoom();
        }
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
            const effectiveIters = this.getCumulativeEffective(maxIdx);
            transitionElem.textContent = `${effectiveIters.toFixed(1)} effective iterations (${currentSegment.precision} phase)`;
        }

        // Update cascading segment datasets with effective iterations
        let cumulativeEffective = 0;
        this.segments.forEach((seg, idx) => {
            const data = [];
            const segStart = seg.start_iteration;
            const segEnd = Math.min(seg.end_iteration, maxIdx);
            const speedup = this.speedupFactors[seg.precision];

            if (maxIdx >= segStart) {
                for (let i = segStart; i <= segEnd; i++) {
                    const point = this.trace[i];
                    if (point) {
                        // Calculate effective x position
                        const iterInSeg = i - segStart;
                        const effectiveX = cumulativeEffective + (iterInSeg / speedup);
                        data.push({
                            x: effectiveX,
                            y: point.residual_norm
                        });
                    }
                }
            }

            this.chart.data.datasets[idx].data = data;

            // Update cumulative for next segment
            if (maxIdx >= seg.end_iteration) {
                cumulativeEffective += seg.iterations / speedup;
            }
        });

        // Update FP64 reference line (if available)
        // FP64 runs at same effective iteration count as cascading for fair comparison
        // Limited to cascading's final effective iterations to show equal compute budget
        if (this.fp64Trace) {
            const fp64DatasetIdx = this.segments.length;
            const currentEffective = this.getCumulativeEffective(maxIdx);
            const cascadeEffectiveFinal = this.getCumulativeEffective(this.trace.length - 1);

            // FP64 iterations = effective iterations (1:1 ratio)
            // Limit FP64 to cascading's effective budget for fair comparison
            const fp64ShowUpTo = Math.min(
                Math.floor(currentEffective),
                Math.floor(cascadeEffectiveFinal),  // Never exceed cascading's effective budget
                this.fp64Trace.length - 1
            );

            const fp64Data = [];
            for (let i = 0; i <= fp64ShowUpTo; i++) {
                const point = this.fp64Trace[i];
                fp64Data.push({
                    x: point.iteration, // FP64 iterations = effective iterations (1:1)
                    y: point.residual_norm
                });
            }
            this.chart.data.datasets[fp64DatasetIdx].data = fp64Data;
        }

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

        // Update comparison summary
        const fp64SummaryElem = document.getElementById('fp64-summary');
        const cascadeSummaryElem = document.getElementById('cascade-summary');
        const speedupSummaryElem = document.getElementById('speedup-summary');

        if (fp64SummaryElem && cascadeSummaryElem && speedupSummaryElem && this.fp64Trace) {
            const effectiveIters = this.getCumulativeEffective(maxIdx);
            const cascadeResidual = this.trace[maxIdx]?.residual_norm;
            const cascadeFinished = maxIdx >= this.trace.length - 1;
            const cascadeEffectiveFinal = this.getCumulativeEffective(this.trace.length - 1);

            // FP64 is limited to cascading's effective budget for fair comparison
            const fp64CurrentIdx = Math.min(
                Math.floor(effectiveIters),
                Math.floor(cascadeEffectiveFinal),
                this.fp64Trace.length - 1
            );

            const fp64CurrentResidual = this.fp64Trace[fp64CurrentIdx]?.residual_norm;
            const cascadeFinalResidual = this.trace[this.trace.length - 1]?.residual_norm;

            if (cascadeResidual && fp64CurrentResidual) {
                // Show cascading progress
                if (cascadeFinished) {
                    cascadeSummaryElem.innerHTML = `<strong>${cascadeEffectiveFinal.toFixed(1)}</strong> eff. iters → <strong>${cascadeFinalResidual.toExponential(2)}</strong> ✓`;
                } else {
                    cascadeSummaryElem.textContent = `${effectiveIters.toFixed(1)} eff. iters → ${cascadeResidual.toExponential(2)}`;
                }

                // Show FP64 progress (same effective iterations budget)
                fp64SummaryElem.textContent = `${fp64CurrentIdx} iters → ${fp64CurrentResidual.toExponential(2)}`;

                // Show comparison at equal compute budget
                if (cascadeFinished) {
                    // Final comparison: same compute budget, different residuals
                    const fp64AtSameBudget = this.fp64Trace[Math.floor(cascadeEffectiveFinal)]?.residual_norm;
                    if (fp64AtSameBudget) {
                        const residualImprovement = fp64AtSameBudget / cascadeFinalResidual;
                        speedupSummaryElem.innerHTML = `<strong>Same budget:</strong> Cascading achieves <span style="color: var(--fp32-color);">${residualImprovement.toExponential(1)}× better</span> residual`;
                    }
                } else {
                    // During animation - show residual comparison at same compute
                    const residualRatio = fp64CurrentResidual / cascadeResidual;
                    speedupSummaryElem.textContent = `At same compute: Cascading ${residualRatio.toFixed(0)}× better residual`;
                }
            }
        }

        this.chart.update();
    }
}

/**
 * Keyboard Navigation Support
 * Provides accessibility shortcuts for visualization controls
 */
function initKeyboardNavigation(vizInstance) {
    document.addEventListener('keydown', (e) => {
        // Don't intercept if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                if (vizInstance.isPlaying) {
                    vizInstance.pause();
                    document.getElementById('playBtn').disabled = false;
                    document.getElementById('pauseBtn').disabled = true;
                } else {
                    vizInstance.play();
                    document.getElementById('playBtn').disabled = true;
                    document.getElementById('pauseBtn').disabled = false;
                }
                break;
            case 'KeyR':
                vizInstance.reset();
                document.getElementById('playBtn').disabled = false;
                document.getElementById('pauseBtn').disabled = true;
                break;
            case 'Escape':
                if (vizInstance.resetZoom) {
                    vizInstance.resetZoom();
                }
                break;
            case 'ArrowRight':
                e.preventDefault();
                vizInstance.currentFrame = Math.min(vizInstance.currentFrame + 10, vizInstance.maxFrames);
                vizInstance.updateVisualization();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                vizInstance.currentFrame = Math.max(vizInstance.currentFrame - 10, 0);
                vizInstance.updateVisualization();
                break;
        }
    });
}

// Export for use in HTML pages
window.PrecisionRaceVisualization = PrecisionRaceVisualization;
window.CascadingPrecisionVisualization = CascadingPrecisionVisualization;
window.initKeyboardNavigation = initKeyboardNavigation;
