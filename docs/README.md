# Precision Lab - GitHub Pages

Interactive visualizations demonstrating precision-performance tradeoffs in numerical computing.

## Files

- **index.html** - Landing page with project overview and key findings
- **race.html** - Precision race visualization comparing FP8/FP16/FP32/FP64
- **cascading.html** - Cascading precision visualization showing FP8→FP16→FP32→FP64
- **styles.css** - Shared styles for all pages
- **js/visualizations.js** - Chart.js-based visualization implementations
- **data/** - JSON trace files from golden test runs

## Local Development

To view locally, serve the docs directory with any static file server:

```bash
# Using Python
cd docs
python -m http.server 8000

# Using Node.js
npx serve docs
```

Then open http://localhost:8000 in your browser.

## Deployment

These files are designed for GitHub Pages deployment. Configure your repository:

1. Go to Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select branch: main, folder: /docs
4. Save

GitHub Pages will automatically serve index.html at your repository URL.

## Dependencies

- **Chart.js 4.4.0** - Loaded from CDN for interactive charts
- **No build step required** - Pure HTML/CSS/JS for maximum compatibility

## Browser Compatibility

- Modern browsers with ES6+ support
- Async/await for data loading
- Canvas API for Chart.js rendering
