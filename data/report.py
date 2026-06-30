"""Generate HTML and PDF reports from data/analyse_dataset.json.

The report is intentionally dependency-free. The HTML report contains the full
tables. The PDF report is a plain text/table rendering written with a minimal
PDF writer from the Python standard library.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable


DEFAULT_INPUT = Path(__file__).resolve().parent / "analyse_dataset.json"
DEFAULT_HTML_OUTPUT = Path(__file__).resolve().parent / "report.html"
DEFAULT_PDF_OUTPUT = Path(__file__).resolve().parent / "report.pdf"


def generate_reports(
    input_path: str | Path = DEFAULT_INPUT,
    html_output_path: str | Path = DEFAULT_HTML_OUTPUT,
    pdf_output_path: str | Path = DEFAULT_PDF_OUTPUT,
) -> None:
    analysis = _load_json(input_path)
    html_report = _build_html_report(analysis)
    pdf_lines = _build_text_report(analysis)

    html_destination = Path(html_output_path).expanduser().resolve()
    pdf_destination = Path(pdf_output_path).expanduser().resolve()
    html_destination.parent.mkdir(parents=True, exist_ok=True)
    pdf_destination.parent.mkdir(parents=True, exist_ok=True)

    html_destination.write_text(html_report, encoding="utf-8")
    _write_pdf(pdf_destination, pdf_lines)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_html_report(analysis: dict[str, Any]) -> str:
    overview = analysis["report_preparation"]["overview"]
    task_detection = analysis["report_preparation"]["task_detection"]
    counts = analysis["counts"]
    evaluation = analysis["evaluation"]
    consistency = evaluation["consistency"]
    intensity = evaluation["intensity"]
    storage = evaluation["storage"]
    memory = evaluation["memory"]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        _html_header(overview, generated_at),
        _section(
            "Dataset Overview",
            _table(
                ["Field", "Value"],
                [
                    ["Dataset name", overview["dataset_name"]],
                    ["Dataset root", overview["dataset_root"]],
                    ["Split", overview["split_filter"]],
                    ["Task type", overview["task_type"]],
                    ["Task reason", overview["task_reason"]],
                    ["Total storage", overview["total_storage"]],
                    ["Minimum memory needed", overview["minimum_memory_needed"]],
                ],
                class_name="kv",
            ),
        ),
        _section(
            "Patient Counts",
            _table(
                ["Metric", "Value"],
                [
                    ["Patients detected", counts["patients_detected"]],
                    ["Patients analysed", counts["patients_analyzed"]],
                    ["Failed analyses", counts["patients_failed"]],
                    ["Patients with missing data", len(evaluation["missing_data"])],
                    ["Annotations present", counts["annotations_present"]],
                    ["Annotations missing", counts["annotations_missing"]],
                    ["Analysis success", f"{overview['analysis_success_percentage']}%"],
                    ["Annotation coverage", f"{overview['annotation_coverage_percentage']}%"],
                ],
                class_name="kv",
            ),
        ),
        _section(
            "Task Detection",
            _table(
                ["Field", "Value"],
                [
                    ["Detected task", task_detection["detected_task"]],
                    ["Reason", task_detection["reason"]],
                    ["Labels found", _yes_no(task_detection["labels_found"])],
                    ["Label files detected", task_detection["label_files_detected"]],
                    [
                        "Image/label pairs detected",
                        task_detection["image_label_pairs_detected"],
                    ],
                ],
                class_name="kv",
            ),
        ),
        _section("Missing Data", _missing_data_table(evaluation["missing_data"])),
        _section(
            "Memory and Storage",
            _table(
                ["Metric", "Value"],
                [
                    ["Total file size", storage["total_file_size_readable"]],
                    ["Images storage", storage["image_file_size_readable"]],
                    ["Labels storage", storage["label_file_size_readable"]],
                    [
                        "Minimum memory needed",
                        memory["minimum_required_memory_readable"],
                    ],
                    [
                        "Minimum memory basis",
                        memory["minimum_required_memory_basis"],
                    ],
                    [
                        "Largest native volume memory",
                        memory["largest_native_volume_readable"],
                    ],
                    [
                        "Largest float32 volume memory",
                        memory["largest_float32_volume_readable"],
                    ],
                ],
                class_name="kv",
            ),
        ),
        _section(
            "Slice Count and Thickness",
            _subsection("Dimensionality", _frequency_table(consistency["dimensionality_frequencies"]))
            + _subsection("Slice Counts", _frequency_table(consistency["slice_count_frequencies"]))
            + _subsection(
                "Slice Thickness",
                _slice_thickness_table(consistency["slice_thickness_frequencies"]),
            )
            + _subsection(
                "Slice Count + Thickness",
                _frequency_table(consistency["slice_count_thickness_frequencies"]),
            ),
        ),
        _section(
            "Voxel Size, Resolution, and Volume",
            _subsection(
                "In-plane Resolution (X x Y mm)",
                _frequency_table(consistency["in_plane_resolution_frequencies"]),
            )
            + _subsection(
                "Voxel Size / Spacing (X x Y x Z mm)",
                _frequency_table(consistency["voxel_spacing_frequencies"]),
            )
            + _subsection(
                "Voxel Volume (mm3)",
                _value_summary_table(consistency["voxel_volume_mm3_summary"]),
            )
            + _subsection(
                "Physical Volume Size (X x Y x Z mm)",
                _frequency_table(consistency["physical_size_frequencies"]),
            )
            + _subsection(
                "Image Dimensions (X x Y x Z voxels)",
                _frequency_table(consistency["dimension_frequencies"]),
            ),
        ),
        _section(
            "Intensity Statistics",
            _subsection(
                "Patient Intensity Summary",
                _dict_table(
                    [
                        ["Global minimum", intensity["global_min"]],
                        ["Global maximum", intensity["global_max"]],
                    ]
                )
                + _intensity_statistics_table(
                    intensity["patient_intensity_statistics_table"]
                ),
            )
            + _subsection(
                "Percentiles",
                _percentile_table(intensity["patient_percentile_table"]),
            )
            + _subsection(
                "Voxel Validity",
                _dict_table(
                    [
                        [
                            "Total voxels",
                            intensity["voxel_validity"]["total_voxels_readable"],
                        ],
                        [
                            "Valid voxels",
                            f"{intensity['voxel_validity']['valid_voxels_readable']} "
                            f"({intensity['voxel_validity']['valid_voxel_percentage']}%)",
                        ],
                        [
                            "Non-finite voxels",
                            f"{intensity['voxel_validity']['non_finite_voxels_readable']} "
                            f"({intensity['voxel_validity']['non_finite_voxel_percentage']}%)",
                        ],
                        ["Explanation", intensity["voxel_validity"]["explanation"]],
                    ]
                ),
            ),
        ),
        _section(
            "Statistics",
            _table(
                ["Metric", "Value"],
                [
                    [
                        "Same in-plane resolution",
                        f"{consistency['percentage_same_resolution']}%",
                    ],
                    ["Same slice count", f"{consistency['percentage_same_slice_count']}%"],
                    [
                        "Same slice thickness",
                        f"{consistency['percentage_same_thickness']}%",
                    ],
                    [
                        "Same voxel spacing",
                        f"{consistency['percentage_same_voxel_spacing']}%",
                    ],
                    [
                        "Different voxel spacing",
                        f"{consistency['percentage_different_voxel_spacing']}%",
                    ],
                    ["Same dimensions", f"{consistency['percentage_same_dimensions']}%"],
                    ["Same orientation", f"{consistency['percentage_same_orientation']}%"],
                    [
                        "Dimensions are consistent",
                        _yes_no(consistency["dimensions_are_consistent"]),
                    ],
                ],
                class_name="kv",
            ),
        ),
        _section("Warnings", _bullet_list(evaluation["warnings"])),
        _section(
            "Preprocessing Recommendations",
            _recommendations_table(evaluation["preprocessing_recommendations"]),
        ),
    ]

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Medical Imaging Dataset Analysis Report</title>",
            _css(),
            "</head>",
            "<body>",
            *sections,
            "</body>",
            "</html>",
        ]
    )


def _html_header(overview: dict[str, Any], generated_at: str) -> str:
    return f"""
<header class="report-header">
  <p class="eyebrow">Medical Imaging Dataset Analysis</p>
  <h1>{_escape(overview["dataset_name"])}</h1>
  <p>Generated on {_escape(generated_at)}</p>
</header>
"""


def _css() -> str:
    return """
<style>
:root {
  --text: #172026;
  --muted: #5f6b73;
  --line: #d8dee3;
  --panel: #f6f8fa;
  --accent: #1f6feb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px;
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  background: white;
  line-height: 1.45;
}
.report-header {
  border-bottom: 2px solid var(--line);
  margin-bottom: 28px;
  padding-bottom: 18px;
}
.eyebrow {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  margin: 0 0 8px;
  text-transform: uppercase;
}
h1 { font-size: 30px; margin: 0 0 8px; }
h2 {
  border-bottom: 1px solid var(--line);
  font-size: 21px;
  margin: 34px 0 14px;
  padding-bottom: 6px;
}
h3 {
  color: #26333b;
  font-size: 16px;
  margin: 20px 0 8px;
}
table {
  border-collapse: collapse;
  margin: 8px 0 18px;
  width: 100%;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--panel);
  font-weight: 700;
}
.kv td:first-child {
  color: var(--muted);
  font-weight: 700;
  width: 28%;
}
ul { margin-top: 8px; }
.empty {
  color: var(--muted);
  font-style: italic;
}
@media print {
  body { padding: 18px; }
  h2 { break-after: avoid; }
  table { break-inside: avoid; }
}
</style>
"""


def _section(title: str, content: str) -> str:
    return f"<section><h2>{_escape(title)}</h2>\n{content}</section>"


def _subsection(title: str, content: str) -> str:
    return f"<h3>{_escape(title)}</h3>\n{content}"


def _table(headers: list[str], rows: Iterable[Iterable[Any]], class_name: str = "") -> str:
    row_list = list(rows)
    if not row_list:
        return '<p class="empty">No data available.</p>'

    class_attr = f' class="{_escape(class_name)}"' if class_name else ""
    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    rows_html = []
    for row in row_list:
        cells = "".join(f"<td>{_escape(_format_value(cell))}</td>" for cell in row)
        rows_html.append(f"<tr>{cells}</tr>")
    return (
        f"<table{class_attr}><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _dict_table(rows: Iterable[Iterable[Any]]) -> str:
    return _table(["Field", "Value"], rows, class_name="kv")


def _value_summary_table(summary: dict[str, Any]) -> str:
    return _table(
        ["Metric", "Value"],
        [
            ["Count", summary["count"]],
            ["Minimum", summary["min"]],
            ["Maximum", summary["max"]],
            ["Mean", summary["mean"]],
            ["Median", summary["median"]],
        ],
        class_name="kv",
    )


def _frequency_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Value", "Patients", "Percentage"],
        (
            [item.get("display_value", item.get("value")), item["count"], f"{item['percentage']}%"]
            for item in items
        ),
    )


def _slice_thickness_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Thickness (mm)", "Patients", "Percentage"],
        (
            [item["thickness_mm"], item["count"], f"{item['percentage']}%"]
            for item in items
        ),
    )


def _missing_data_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Patient ID", "Split", "Missing data", "Image", "Label", "Reason"],
        (
            [
                item["patient_id"],
                item["split"],
                ", ".join(item["missing_fields"]),
                item["image_path"],
                item["label_path"] or "Not available",
                item["reason"],
            ]
            for item in items
        ),
    )


def _intensity_statistics_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Statistic", "Unit", "Patients", "Minimum", "Maximum", "Mean", "Median"],
        (
            [
                item["statistic"],
                item["unit"],
                item["count"],
                item["min"],
                item["max"],
                item["mean"],
                item["median"],
            ]
            for item in items
        ),
    )


def _percentile_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Percentile", "Patients", "Minimum", "Maximum", "Mean", "Median"],
        (
            [
                item["percentile"],
                item["count"],
                item["min"],
                item["max"],
                item["mean"],
                item["median"],
            ]
            for item in items
        ),
    )


def _recommendations_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Severity", "Category", "Recommendation", "Reason"],
        (
            [
                item["severity"],
                item["category"],
                item["message"],
                item["reason"],
            ]
            for item in items
        ),
    )


def _bullet_list(items: list[str]) -> str:
    if not items:
        return '<p class="empty">No warnings.</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _build_text_report(analysis: dict[str, Any]) -> list[str]:
    overview = analysis["report_preparation"]["overview"]
    task_detection = analysis["report_preparation"]["task_detection"]
    counts = analysis["counts"]
    evaluation = analysis["evaluation"]
    consistency = evaluation["consistency"]
    intensity = evaluation["intensity"]
    memory = evaluation["memory"]

    lines: list[str] = []
    _add_title(lines, "Medical Imaging Dataset Analysis Report")
    _add_lines(
        lines,
        [
            f"Dataset: {overview['dataset_name']}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Task: {overview['task_type']} ({overview['task_reason']})",
        ],
    )

    _add_title(lines, "Dataset Overview")
    _add_key_values(
        lines,
        [
            ("Dataset root", overview["dataset_root"]),
            ("Split", overview["split_filter"]),
            ("Patients detected", counts["patients_detected"]),
            ("Patients analysed", counts["patients_analyzed"]),
            ("Failed analyses", counts["patients_failed"]),
            ("Patients with missing data", len(evaluation["missing_data"])),
            ("Total storage", overview["total_storage"]),
            ("Minimum memory", overview["minimum_memory_needed"]),
        ],
    )

    _add_title(lines, "Task Detection")
    _add_key_values(
        lines,
        [
            ("Detected task", task_detection["detected_task"]),
            ("Reason", task_detection["reason"]),
            ("Labels found", _yes_no(task_detection["labels_found"])),
            ("Label files detected", task_detection["label_files_detected"]),
            ("Image/label pairs", task_detection["image_label_pairs_detected"]),
        ],
    )

    _add_title(lines, "Missing Data")
    _add_table_lines(
        lines,
        ["Patient ID", "Split", "Missing data"],
        (
            [
                item["patient_id"],
                item["split"],
                ", ".join(item["missing_fields"]),
            ]
            for item in evaluation["missing_data"]
        ),
    )

    _add_title(lines, "Memory")
    _add_key_values(
        lines,
        [
            ("Minimum memory needed", memory["minimum_required_memory_readable"]),
            ("Basis", memory["minimum_required_memory_basis"]),
            ("Largest native volume", memory["largest_native_volume_readable"]),
            ("Largest float32 volume", memory["largest_float32_volume_readable"]),
        ],
    )

    _add_title(lines, "Slices")
    _add_table_lines(
        lines,
        ["Dimensionality", "Patients", "%"],
        _frequency_text_rows(consistency["dimensionality_frequencies"]),
    )
    _add_table_lines(
        lines,
        ["Slice count", "Patients", "%"],
        _frequency_text_rows(consistency["slice_count_frequencies"]),
    )
    _add_table_lines(
        lines,
        ["Thickness (mm)", "Patients", "%"],
        (
            [item["thickness_mm"], item["count"], item["percentage"]]
            for item in consistency["slice_thickness_frequencies"]
        ),
    )

    _add_title(lines, "Voxel Size and Volume")
    _add_table_lines(
        lines,
        ["Voxel size XxYxZ (mm)", "Patients", "%"],
        _frequency_text_rows(consistency["voxel_spacing_frequencies"]),
    )
    _add_table_lines(
        lines,
        ["Physical volume XxYxZ (mm)", "Patients", "%"],
        _frequency_text_rows(consistency["physical_size_frequencies"]),
    )
    _add_table_lines(
        lines,
        ["Dimensions XxYxZ", "Patients", "%"],
        _frequency_text_rows(consistency["dimension_frequencies"]),
    )

    _add_title(lines, "Intensity")
    _add_table_lines(
        lines,
        ["Statistic", "Unit", "Min", "Max", "Mean", "Median"],
        (
            [
                item["statistic"],
                item["unit"],
                item["min"],
                item["max"],
                item["mean"],
                item["median"],
            ]
            for item in intensity["patient_intensity_statistics_table"]
        ),
    )
    _add_table_lines(
        lines,
        ["Percentile", "Min", "Max", "Mean", "Median"],
        (
            [
                item["percentile"],
                item["min"],
                item["max"],
                item["mean"],
                item["median"],
            ]
            for item in intensity["patient_percentile_table"]
        ),
    )
    _add_key_values(
        lines,
        [
            ("Total voxels", intensity["voxel_validity"]["total_voxels_readable"]),
            ("Valid voxels", intensity["voxel_validity"]["valid_voxels_readable"]),
            (
                "Non-finite voxels",
                intensity["voxel_validity"]["non_finite_voxels_readable"],
            ),
        ],
    )

    _add_title(lines, "Warnings")
    _add_lines(lines, evaluation["warnings"] or ["No warnings."])

    _add_title(lines, "Preprocessing Recommendations")
    _add_table_lines(
        lines,
        ["Severity", "Category", "Recommendation"],
        (
            [item["severity"], item["category"], item["message"]]
            for item in evaluation["preprocessing_recommendations"]
        ),
    )

    return lines


def _frequency_text_rows(items: list[dict[str, Any]]) -> Iterable[list[Any]]:
    for item in items:
        yield [item.get("display_value", item.get("value")), item["count"], item["percentage"]]


def _add_title(lines: list[str], title: str) -> None:
    lines.extend(["", title, "=" * len(title)])


def _add_lines(lines: list[str], values: Iterable[Any]) -> None:
    for value in values:
        lines.append(str(value))


def _add_key_values(lines: list[str], rows: Iterable[tuple[str, Any]]) -> None:
    for key, value in rows:
        lines.append(f"{key}: {_format_value(value)}")


def _add_table_lines(
    lines: list[str],
    headers: list[str],
    rows: Iterable[Iterable[Any]],
) -> None:
    row_list = [[_format_value(cell) for cell in row] for row in rows]
    if not row_list:
        lines.append("No data available.")
        return

    normalized_headers = [_format_value(header) for header in headers]
    widths = [
        min(
            max(
                len(normalized_headers[index]),
                *(len(row[index]) for row in row_list),
            ),
            34,
        )
        for index in range(len(normalized_headers))
    ]
    lines.append(_format_table_row(normalized_headers, widths))
    lines.append(_format_table_row(["-" * width for width in widths], widths))
    for row in row_list:
        lines.append(_format_table_row(row, widths))


def _format_table_row(row: list[str], widths: list[int]) -> str:
    cells = []
    for value, width in zip(row, widths, strict=True):
        if len(value) > width:
            value = value[: width - 1] + "."
        cells.append(value.ljust(width))
    return " | ".join(cells)


def _write_pdf(path: Path, lines: list[str]) -> None:
    pages = _paginate_pdf_lines(lines)
    objects: list[bytes] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")  # Pages object filled after page objects are known.
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_object_numbers: list[int] = []
    for page_lines in pages:
        page_object_number = len(objects) + 1
        content_object_number = page_object_number + 1
        page_object_numbers.append(page_object_number)

        content = _pdf_page_content(page_lines)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(
        "latin-1"
    )

    _write_pdf_objects(path, objects)


def _paginate_pdf_lines(lines: list[str], max_chars: int = 98, max_lines: int = 58) -> list[list[str]]:
    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped = wrap(line, width=max_chars, replace_whitespace=False) or [""]
        wrapped_lines.extend(wrapped)

    return [
        wrapped_lines[index : index + max_lines]
        for index in range(0, len(wrapped_lines), max_lines)
    ] or [["No report content."]]


def _pdf_page_content(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 9 Tf", "40 805 Td", "12 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray()
    content.extend(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(content)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return _yes_no(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return " x ".join(_format_value(item) for item in value)
    return str(value)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _escape(value: Any) -> str:
    return html.escape(_format_value(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate HTML and PDF reports from analyse_dataset.json."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input dataset analysis JSON path.",
    )
    parser.add_argument(
        "--html",
        default=str(DEFAULT_HTML_OUTPUT),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF_OUTPUT),
        help="Output PDF report path.",
    )
    args = parser.parse_args(argv)

    generate_reports(args.input, args.html, args.pdf)
    print(f"HTML report written to: {Path(args.html).resolve()}")
    print(f"PDF report written to: {Path(args.pdf).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
