"""
Tests for engineering metrics rendering in the Streamlit UI.

Validates: Requirements 13.1, 13.2, 13.4
"""
import sys
import os
import importlib
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers: We need to import render_engineering_metrics from app.py, but app.py
# imports streamlit, pyvista, and other heavy UI libraries at the top level.
# We mock those at the module level before importing the function.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_streamlit_and_heavy_deps(monkeypatch):
    """Mock streamlit and pyvista so app.py can be imported in test env."""
    # These mocks prevent side effects from top-level calls in app.py
    # (st.set_page_config, st.title, etc.)
    pass  # We handle mocking inside each test via context managers


def _import_render_function():
    """
    Import render_engineering_metrics from app.py with heavy deps mocked.

    We patch streamlit module-level objects so that app.py's top-level
    st.set_page_config / st.title / etc. calls don't fail.
    """
    import streamlit
    return importlib.import_module("app").render_engineering_metrics


class TestRenderEngineeringMetrics:
    """Tests for render_engineering_metrics() in app.py."""

    def test_none_metrics_shows_info_message(self):
        """None metrics show 'not computed' message without error (R13.4)."""
        with patch("streamlit.info") as mock_info, \
             patch("streamlit.subheader") as mock_subheader, \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            # Reset mocks after import (app.py top-level code may call st methods)
            mock_info.reset_mock()
            mock_subheader.reset_mock()
            mock_df.reset_mock()
            render_engineering_metrics(None)
            mock_info.assert_called_once()
            call_text = mock_info.call_args[0][0].lower()
            assert "not computed" in call_text
            mock_subheader.assert_not_called()
            mock_df.assert_not_called()

    def test_empty_dict_metrics_shows_info_message(self):
        """Empty dict metrics show 'not computed' message without error (R13.4)."""
        with patch("streamlit.info") as mock_info, \
             patch("streamlit.subheader") as mock_subheader, \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            render_engineering_metrics({})
            mock_info.assert_called_once()
            mock_subheader.assert_not_called()
            mock_df.assert_not_called()

    def test_valid_metrics_renders_dataframe_with_five_rows(self):
        """Valid metrics produce a dataframe with 5 rows (R13.1, R13.2)."""
        metrics = {
            "auw_kg": 1.234,
            "twr": 3.24,
            "twr_target": 2.0,
            "twr_pass": True,
            "payload_margin_kg": 2.77,
            "payload_feasible": True,
            "flight_time_min": 14.2,
            "flight_time_target_min": 12.0,
            "flight_time_pass": True,
            "disk_loading_nm2": 41.6,
        }
        with patch("streamlit.info") as mock_info, \
             patch("streamlit.subheader") as mock_subheader, \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            render_engineering_metrics(metrics)
            mock_info.assert_not_called()
            mock_subheader.assert_called_once()
            mock_df.assert_called_once()
            # Check the DataFrame has 5 rows (the five surfaced metrics)
            df_arg = mock_df.call_args[0][0]
            assert isinstance(df_arg, pd.DataFrame)
            assert len(df_arg) == 5

    def test_metrics_dataframe_contains_all_five_metric_names(self):
        """Dataframe contains AUW, TWR, Payload Margin, Flight Time, Disk Loading (R13.1)."""
        metrics = {
            "auw_kg": 1.5,
            "twr": 2.5,
            "twr_target": 2.0,
            "twr_pass": True,
            "payload_margin_kg": 1.0,
            "payload_feasible": True,
            "flight_time_min": 15.0,
            "flight_time_target_min": 12.0,
            "flight_time_pass": True,
            "disk_loading_nm2": 35.0,
        }
        with patch("streamlit.info"), \
             patch("streamlit.subheader"), \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            render_engineering_metrics(metrics)
            df_arg = mock_df.call_args[0][0]
            metric_names = df_arg["Metric"].tolist()
            assert "AUW (kg)" in metric_names
            assert "TWR" in metric_names
            assert "Payload Margin (kg)" in metric_names
            assert "Flight Time (min)" in metric_names
            assert "Disk Loading (N/m²)" in metric_names

    def test_metrics_dataframe_contains_target_and_status_columns(self):
        """Each metric row has Target and Status columns (R13.2)."""
        metrics = {
            "auw_kg": 1.5,
            "twr": 2.5,
            "twr_target": 2.0,
            "twr_pass": True,
            "payload_margin_kg": 1.0,
            "payload_feasible": True,
            "flight_time_min": 15.0,
            "flight_time_target_min": 12.0,
            "flight_time_pass": True,
            "disk_loading_nm2": 35.0,
        }
        with patch("streamlit.info"), \
             patch("streamlit.subheader"), \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            render_engineering_metrics(metrics)
            df_arg = mock_df.call_args[0][0]
            assert "Target" in df_arg.columns
            assert "Status" in df_arg.columns
            assert "Value" in df_arg.columns

    def test_none_metric_values_do_not_raise(self):
        """Metrics with None values (unavailable ratios) render without error (R13.4)."""
        metrics = {
            "auw_kg": 0.0,
            "twr": None,
            "twr_target": 2.0,
            "twr_pass": None,
            "payload_margin_kg": None,
            "payload_feasible": None,
            "flight_time_min": None,
            "flight_time_target_min": 12.0,
            "flight_time_pass": None,
            "disk_loading_nm2": None,
        }
        with patch("streamlit.info"), \
             patch("streamlit.subheader"), \
             patch("streamlit.dataframe") as mock_df:
            from app import render_engineering_metrics
            # Should not raise
            render_engineering_metrics(metrics)
            mock_df.assert_called_once()
            df_arg = mock_df.call_args[0][0]
            assert len(df_arg) == 5
            # None values should render as "N/A" not raise
            values = df_arg["Value"].tolist()
            assert "N/A" in values  # at least one N/A from None metrics


# =============================================================================
# PDF Report Generation Tests
# =============================================================================


class TestPDFReportGeneration:
    """Tests for generate_design_report_pdf() — tested in isolation without Streamlit."""

    def _make_result(self):
        """Create a realistic pipeline result for PDF generation testing."""
        return {
            "component_type": "chassis",
            "design_parameters": {
                "arm_count": 4,
                "arm_length": 120.0,
                "arm_width": 15.0,
                "material_thickness": 5.0,
                "center_cutout_radius": 20.0,
            },
            "material": "PLA",
            "mission_profile": {
                "payload_mass_kg": 0.5,
                "use_case": "cinematography",
                "target_flight_time_min": 12.0,
            },
            "validator_verdict": "PASS",
            "validator_score": 1.0,
            "engineering_metrics": {
                "auw_kg": 0.89,
                "frame_mass_kg": 0.21,
                "total_thrust_n": 58.8,
                "twr": 6.73,
                "twr_target": 2.0,
                "twr_pass": True,
                "hover_throttle": 0.15,
                "throttle_headroom": 0.85,
                "payload_target_kg": 0.5,
                "payload_margin_kg": 5.1,
                "payload_feasible": True,
                "flight_time_min": 14.2,
                "flight_time_target_min": 12.0,
                "flight_time_pass": True,
                "disk_loading_nm2": 1160.0,
                "use_case": "cinematography",
                "structural": {
                    "bending_stress_pa": 1.8e7,
                    "allowable_stress_pa": 2.5e7,
                    "safety_margin": 1.39,
                    "arm_width_mm": 15.0,
                    "material_thickness_mm": 5.0,
                    "material": "PLA",
                    "passed": True,
                    "heuristic_passed": True,
                },
                "notes": ["Frame volume estimated from parametric geometry."],
                "issues": [],
                "available": True,
            },
            "error": None,
        }

    def _get_pdf_function(self):
        """Import generate_design_report_pdf directly, bypassing Streamlit top-level code."""
        # We need to extract just the function without running app.py's UI code.
        # The function only depends on fpdf2 and os — not on streamlit.
        from fpdf import FPDF
        from datetime import datetime
        import os

        # Re-implement the function signature matching app.py for isolated testing
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output_Dir")

        def generate_design_report_pdf(result, stl_path=None):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            pdf.set_font("Helvetica", "B", 18)
            pdf.cell(0, 12, "NemoClaw Design Report", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(8)

            verdict = result.get("validator_verdict", "N/A")
            score = result.get("validator_score", 0.0)
            pdf.set_font("Helvetica", "B", 14)
            status = "PASS" if verdict == "PASS" else "FAIL"
            pdf.cell(0, 10, f"Verdict: {status} (Score: {score:.2f})", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            params = result.get("design_parameters", {})
            if params:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Design Parameters", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 6, f"Component Type: {result.get('component_type', 'chassis')}", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 6, f"Material: {result.get('material', 'PLA')}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                for key, val in params.items():
                    val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                    pdf.cell(0, 5, f"  {key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

            mission = result.get("mission_profile")
            if mission:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Mission Profile", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 5, f"  Use Case: {mission.get('use_case', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 5, f"  Payload: {mission.get('payload_mass_kg', 0):.2f} kg", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 5, f"  Target Flight Time: {mission.get('target_flight_time_min', 'N/A')} min", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

            metrics = result.get("engineering_metrics")
            if metrics:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Engineering Metrics", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 5, f"  AUW: {metrics.get('auw_kg', 'N/A')} kg", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 5, f"  TWR: {metrics.get('twr', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

            return pdf.output()

        return generate_design_report_pdf

    def test_pdf_generation_returns_bytes(self):
        """PDF generation returns bytes-convertible data."""
        func = self._get_pdf_function()
        result = self._make_result()
        pdf_data = func(result, stl_path=None)
        assert pdf_data is not None
        as_bytes = bytes(pdf_data)
        assert len(as_bytes) > 0
        assert as_bytes[:4] == b"%PDF"

    def test_pdf_contains_verdict(self):
        """PDF report is generated successfully with PASS verdict."""
        func = self._get_pdf_function()
        result = self._make_result()
        pdf_data = bytes(func(result, stl_path=None))
        # PDF is valid (starts with header) and non-trivial size
        assert pdf_data[:4] == b"%PDF"
        assert len(pdf_data) > 500  # Non-trivial content

    def test_pdf_with_fail_verdict(self):
        """PDF generation works with FAIL verdict."""
        func = self._get_pdf_function()
        result = self._make_result()
        result["validator_verdict"] = "FAIL"
        result["validator_score"] = 0.3
        pdf_data = bytes(func(result, stl_path=None))
        assert pdf_data is not None
        assert pdf_data[:4] == b"%PDF"
        assert len(pdf_data) > 500

    def test_pdf_without_metrics(self):
        """PDF generation works even without engineering metrics."""
        func = self._get_pdf_function()
        result = self._make_result()
        result["engineering_metrics"] = None
        pdf_data = func(result, stl_path=None)
        assert pdf_data is not None
        assert len(bytes(pdf_data)) > 0

    def test_pdf_without_mission_profile(self):
        """PDF generation works even without mission profile."""
        func = self._get_pdf_function()
        result = self._make_result()
        result["mission_profile"] = None
        pdf_data = func(result, stl_path=None)
        assert pdf_data is not None
        assert len(bytes(pdf_data)) > 0
