"""Tests for visualization HTML pages and associated assets.

These tests verify that the visualization pages and data files are properly
structured and can be loaded correctly.
"""

import json
import re
from pathlib import Path

import pytest

# Paths to docs directory
DOCS_DIR = Path(__file__).parent.parent / "docs"


class TestHTMLPages:
    """Test HTML page structure and validity."""

    @pytest.fixture
    def cascading_html(self) -> str:
        """Load cascading.html content."""
        return (DOCS_DIR / "cascading.html").read_text()

    @pytest.fixture
    def race_html(self) -> str:
        """Load race.html content."""
        return (DOCS_DIR / "race.html").read_text()

    def test_cascading_html_exists(self) -> None:
        """Verify cascading.html exists."""
        assert (DOCS_DIR / "cascading.html").exists()

    def test_race_html_exists(self) -> None:
        """Verify race.html exists."""
        assert (DOCS_DIR / "race.html").exists()

    def test_cascading_html_has_required_elements(self, cascading_html: str) -> None:
        """Verify cascading.html contains required HTML elements."""
        # Required elements
        assert "<canvas id=" in cascading_html, "Missing chart canvas"
        assert "chart-canvas" in cascading_html, "Missing chart-canvas id"
        assert 'id="playBtn"' in cascading_html, "Missing play button"
        assert 'id="pauseBtn"' in cascading_html, "Missing pause button"
        assert 'id="resetBtn"' in cascading_html, "Missing reset button"

    def test_race_html_has_required_elements(self, race_html: str) -> None:
        """Verify race.html contains required HTML elements."""
        # Required elements
        assert "<canvas id=" in race_html, "Missing chart canvas"
        assert "chart-canvas" in race_html, "Missing chart-canvas id"
        assert 'id="playBtn"' in race_html, "Missing play button"
        assert 'id="pauseBtn"' in race_html, "Missing pause button"
        assert 'id="resetBtn"' in race_html, "Missing reset button"

    def test_cascading_html_loads_chart_js(self, cascading_html: str) -> None:
        """Verify cascading.html loads Chart.js library."""
        assert "chart.js" in cascading_html.lower(), "Missing Chart.js script"

    def test_race_html_loads_chart_js(self, race_html: str) -> None:
        """Verify race.html loads Chart.js library."""
        assert "chart.js" in race_html.lower(), "Missing Chart.js script"

    def test_cascading_html_loads_visualization_js(self, cascading_html: str) -> None:
        """Verify cascading.html loads visualizations.js."""
        assert "visualizations.js" in cascading_html, "Missing visualizations.js script"

    def test_race_html_loads_visualization_js(self, race_html: str) -> None:
        """Verify race.html loads visualizations.js."""
        assert "visualizations.js" in race_html, "Missing visualizations.js script"

    def test_html_sri_hashes_valid_format(self, cascading_html: str) -> None:
        """Verify SRI hashes are in valid format (sha384-base64)."""
        sri_pattern = r'integrity="sha384-[A-Za-z0-9+/=]+"'
        matches = re.findall(sri_pattern, cascading_html)
        # Should have 3 SRI hashes (chart.js, hammer.js, chartjs-plugin-zoom)
        assert (
            len(matches) >= 3
        ), f"Expected at least 3 SRI hashes, found {len(matches)}"

    def test_cascading_html_uses_cascade_trace_data(self, cascading_html: str) -> None:
        """Verify cascading.html references cascade trace data file."""
        assert (
            "cascade_trace.json" in cascading_html
        ), "Missing cascade_trace.json reference"

    def test_race_html_uses_precision_trace_data(self, race_html: str) -> None:
        """Verify race.html references all precision trace data files."""
        assert (
            "fp8_e4m3_trace.json" in race_html
        ), "Missing fp8_e4m3_trace.json reference"
        assert "fp16_trace.json" in race_html, "Missing fp16_trace.json reference"
        assert "fp32_trace.json" in race_html, "Missing fp32_trace.json reference"
        assert "fp64_trace.json" in race_html, "Missing fp64_trace.json reference"


class TestTraceDataFiles:
    """Test trace data JSON files structure and validity."""

    @pytest.fixture
    def cascade_trace(self) -> dict:
        """Load cascade trace data."""
        return json.loads((DOCS_DIR / "data" / "cascade_trace.json").read_text())

    @pytest.fixture
    def fp64_trace(self) -> dict:
        """Load FP64 trace data."""
        return json.loads((DOCS_DIR / "data" / "fp64_trace.json").read_text())

    @pytest.fixture
    def fp32_trace(self) -> dict:
        """Load FP32 trace data."""
        return json.loads((DOCS_DIR / "data" / "fp32_trace.json").read_text())

    @pytest.fixture
    def fp16_trace(self) -> dict:
        """Load FP16 trace data."""
        return json.loads((DOCS_DIR / "data" / "fp16_trace.json").read_text())

    @pytest.fixture
    def fp8_trace(self) -> dict:
        """Load FP8 trace data."""
        return json.loads((DOCS_DIR / "data" / "fp8_e4m3_trace.json").read_text())

    def test_data_directory_exists(self) -> None:
        """Verify data directory exists."""
        assert (DOCS_DIR / "data").is_dir()

    def test_cascade_trace_exists(self) -> None:
        """Verify cascade_trace.json exists."""
        assert (DOCS_DIR / "data" / "cascade_trace.json").exists()

    def test_all_precision_traces_exist(self) -> None:
        """Verify all precision trace files exist."""
        assert (DOCS_DIR / "data" / "fp8_e4m3_trace.json").exists()
        assert (DOCS_DIR / "data" / "fp16_trace.json").exists()
        assert (DOCS_DIR / "data" / "fp32_trace.json").exists()
        assert (DOCS_DIR / "data" / "fp64_trace.json").exists()

    def test_cascade_trace_has_required_fields(self, cascade_trace: dict) -> None:
        """Verify cascade trace has required structure."""
        assert "metadata" in cascade_trace, "Missing metadata field"
        assert "segments" in cascade_trace, "Missing segments field"
        assert isinstance(cascade_trace["segments"], list), "Segments should be a list"
        assert len(cascade_trace["segments"]) > 0, "Segments should not be empty"

    def test_cascade_trace_metadata_fields(self, cascade_trace: dict) -> None:
        """Verify cascade trace metadata has expected fields."""
        meta = cascade_trace["metadata"]
        required_fields = [
            "matrix_size",
            "condition_number",
            "seed",
            "convergence_type",
        ]
        for field in required_fields:
            assert field in meta, f"Missing metadata field: {field}"

    def test_cascade_trace_segment_structure(self, cascade_trace: dict) -> None:
        """Verify cascade trace segments have expected structure."""
        segment = cascade_trace["segments"][0]
        required_fields = ["precision", "start_iteration", "end_iteration"]
        for field in required_fields:
            assert field in segment, f"Missing segment field: {field}"

    def test_precision_trace_has_required_fields(self, fp64_trace: dict) -> None:
        """Verify precision trace has required structure."""
        assert "metadata" in fp64_trace, "Missing metadata field"
        assert "trace" in fp64_trace, "Missing trace field"
        assert isinstance(fp64_trace["trace"], list), "Trace should be a list"
        assert len(fp64_trace["trace"]) > 0, "Trace should not be empty"

    def test_precision_trace_entry_structure(self, fp64_trace: dict) -> None:
        """Verify trace entries have expected structure."""
        entry = fp64_trace["trace"][0]
        required_fields = ["iteration", "residual_norm"]
        for field in required_fields:
            assert field in entry, f"Missing trace entry field: {field}"

    def test_all_precision_traces_valid_json(self) -> None:
        """Verify all trace files are valid JSON."""
        data_dir = DOCS_DIR / "data"
        for json_file in data_dir.glob("*.json"):
            try:
                json.loads(json_file.read_text())
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file.name}: {e}")

    def test_fp8_trace_residual_floor(self, fp8_trace: dict) -> None:
        """Verify FP8 trace shows expected precision floor behavior."""
        trace = fp8_trace["trace"]
        final_residual = trace[-1]["residual_norm"]
        # FP8 should plateau around 1e-2 (never reach 1e-7)
        assert final_residual > 1e-5, f"FP8 residual {final_residual} unexpectedly low"

    def test_fp64_trace_residual_floor(self, fp64_trace: dict) -> None:
        """Verify FP64 trace reaches expected precision."""
        trace = fp64_trace["trace"]
        final_residual = trace[-1]["residual_norm"]
        # FP64 should reach at least 1e-10
        assert (
            final_residual < 1e-10
        ), f"FP64 residual {final_residual} unexpectedly high"

    def test_traces_use_same_matrix(
        self, fp8_trace: dict, fp16_trace: dict, fp32_trace: dict, fp64_trace: dict
    ) -> None:
        """Verify all traces used the same test matrix (same seed)."""
        seeds = {
            fp8_trace["metadata"]["seed"],
            fp16_trace["metadata"]["seed"],
            fp32_trace["metadata"]["seed"],
            fp64_trace["metadata"]["seed"],
        }
        assert len(seeds) == 1, f"Traces use different seeds: {seeds}"


class TestVisualizationJS:
    """Test visualization JavaScript file."""

    @pytest.fixture
    def viz_js(self) -> str:
        """Load visualizations.js content."""
        return (DOCS_DIR / "js" / "visualizations.js").read_text()

    def test_visualization_js_exists(self) -> None:
        """Verify visualizations.js exists."""
        assert (DOCS_DIR / "js" / "visualizations.js").exists()

    def test_has_animated_visualization_class(self, viz_js: str) -> None:
        """Verify AnimatedVisualization class is defined."""
        assert "class AnimatedVisualization" in viz_js

    def test_has_precision_race_visualization_class(self, viz_js: str) -> None:
        """Verify PrecisionRaceVisualization class is defined."""
        assert "class PrecisionRaceVisualization" in viz_js

    def test_has_cascading_visualization_class(self, viz_js: str) -> None:
        """Verify CascadingPrecisionVisualization class is defined."""
        assert "class CascadingPrecisionVisualization" in viz_js

    def test_has_color_definitions(self, viz_js: str) -> None:
        """Verify color constants are defined."""
        assert "COLORS" in viz_js
        assert "fp8" in viz_js
        assert "fp16" in viz_js
        assert "fp32" in viz_js
        assert "fp64" in viz_js

    def test_has_keyboard_navigation(self, viz_js: str) -> None:
        """Verify keyboard navigation function is defined."""
        assert "initKeyboardNavigation" in viz_js


class TestStylesCSS:
    """Test CSS styles file."""

    def test_styles_css_exists(self) -> None:
        """Verify styles.css exists."""
        assert (DOCS_DIR / "styles.css").exists()

    def test_styles_has_precision_colors(self) -> None:
        """Verify CSS has precision color definitions."""
        css = (DOCS_DIR / "styles.css").read_text()
        assert "--fp8-color" in css, "Missing FP8 color variable"
        assert "--fp16-color" in css, "Missing FP16 color variable"
        assert "--fp32-color" in css, "Missing FP32 color variable"
        assert "--fp64-color" in css, "Missing FP64 color variable"
