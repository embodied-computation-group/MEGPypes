from __future__ import annotations

import base64
import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

# Editable QC plot definitions. Order in this list is display order.
QC_PLOT_ORDER: list[dict[str, str]] = [
	{
		"file_hint": "desc-psd_before_zapline",
		"title": "Power Spectral Density Before Zapline",
	},
	{
		"file_hint": "desc-ica_components",
		"title": "ICA Components",
	},
	{
		"file_hint": "desc-raw_epochs",
		"title": "Raw Epochs",
	},
	{}
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


@dataclass(frozen=True)
class SessionInfo:
	subject: str
	session: str
	session_dir: Path
	qc_files: list[Path]

	@property
	def key(self) -> str:
		return f"{self.subject} | {self.session}"


def _looks_like_subject_dir(path: Path) -> bool:
	return path.is_dir() and path.name.startswith("sub-")


def _looks_like_session_dir(path: Path) -> bool:
	return path.is_dir() and path.name.startswith("ses-")


def _collect_qc_files(session_dir: Path) -> list[Path]:
	# Gather images from any qc-like folder inside the session to be robust to variants.
	qc_dirs = [p for p in session_dir.rglob("*") if p.is_dir() and p.name.lower() == "qc"]
	files: list[Path] = []
	for qc_dir in qc_dirs:
		files.extend(
			[p for p in qc_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
		)
	return sorted(set(files))


@st.cache_data(show_spinner=False)
def discover_sessions(root_dir: str) -> list[SessionInfo]:
	root = Path(root_dir).expanduser().resolve()
	sessions: list[SessionInfo] = []

	if not root.exists():
		return sessions

	for subject_dir in sorted(root.rglob("sub-*")):
		if not _looks_like_subject_dir(subject_dir):
			continue
		for session_dir in sorted(subject_dir.glob("ses-*")):
			if not _looks_like_session_dir(session_dir):
				continue
			qc_files = _collect_qc_files(session_dir)
			if not qc_files:
				continue
			sessions.append(
				SessionInfo(
					subject=subject_dir.name,
					session=session_dir.name,
					session_dir=session_dir,
					qc_files=qc_files,
				)
			)

	sessions.sort(key=lambda item: (item.subject, item.session))
	return sessions


def sort_files_by_qc_order(files: list[Path], qc_order: list[dict[str, str]]) -> list[tuple[str, Path]]:
	used: set[Path] = set()
	ordered: list[tuple[str, Path]] = []

	for rule in qc_order:
		hint = rule.get("file_hint", "").strip()
		title = rule.get("title", hint).strip() or hint
		if not hint:
			continue

		matches = [path for path in files if hint in path.name and path not in used]
		for match in sorted(matches):
			ordered.append((title, match))
			used.add(match)

	for remaining in sorted([path for path in files if path not in used]):
		ordered.append((remaining.stem, remaining))

	return ordered


def _img_to_data_uri(image_path: Path) -> str:
	mime_by_suffix = {
		".png": "image/png",
		".jpg": "image/jpeg",
		".jpeg": "image/jpeg",
		".webp": "image/webp",
		".svg": "image/svg+xml",
	}
	mime = mime_by_suffix.get(image_path.suffix.lower(), "application/octet-stream")
	encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
	return f"data:{mime};base64,{encoded}"


def build_report_html(
	sessions: list[SessionInfo],
	qc_order: list[dict[str, str]],
	title: str = "MEG QC Report",
) -> str:
	blocks: list[str] = []

	for session in sessions:
		ordered_files = sort_files_by_qc_order(session.qc_files, qc_order)
		image_sections: list[str] = []
		for image_title, image_path in ordered_files:
			data_uri = _img_to_data_uri(image_path)
			image_sections.append(
				"\n".join(
					[
						"<section class='plot-card'>",
						f"<h3>{html.escape(image_title)}</h3>",
						f"<img src='{data_uri}' alt='{html.escape(image_title)}' />",
						f"<p class='path'>{html.escape(str(image_path))}</p>",
						"</section>",
					]
				)
			)

		blocks.append(
			"\n".join(
				[
					"<article class='session-block'>",
					f"<h2>{html.escape(session.subject)} - {html.escape(session.session)}</h2>",
					*image_sections,
					"</article>",
				]
			)
		)

	return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
	body {{
	  margin: 0;
	  padding: 1.5rem;
	  font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
	  background: #f4f7fb;
	  color: #1b2a38;
	}}
	h1 {{ margin-bottom: 1.25rem; }}
	.session-block {{
	  background: white;
	  border: 1px solid #dbe6f2;
	  border-radius: 0.75rem;
	  padding: 1rem;
	  margin-bottom: 1rem;
	}}
	.plot-card {{
	  margin-top: 1rem;
	  padding-top: 1rem;
	  border-top: 1px solid #edf2f7;
	  page-break-inside: avoid;
	}}
	.plot-card img {{
	  width: 100%;
	  height: 420px;
	  border: 1px solid #dbe6f2;
	  border-radius: 0.5rem;
	  background: #fff;
	  object-fit: contain;
	}}
	.path {{
	  font-size: 0.78rem;
	  color: #64748b;
	  word-break: break-word;
	}}
	@media print {{
	  body {{ background: white; padding: 0; }}
	  .session-block {{ border: none; }}
	}}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {content}
</body>
</html>
""".format(title=html.escape(title), content="\n".join(blocks))


def _default_root() -> str:
	env_root = os.environ.get("MEGPYPES_REPORT_ROOT")
	if env_root:
		return str(Path(env_root).expanduser().resolve())

	workspace = Path.cwd()
	candidates = [
		workspace / "workdir" / "megpreproc" / "output",
		workspace,
	]
	for candidate in candidates:
		if candidate.exists():
			return str(candidate)
	return str(workspace)


def _ensure_state(session_count: int) -> None:
	if "selected_session_idx" not in st.session_state:
		st.session_state.selected_session_idx = 0
	if "selected_plot_idx" not in st.session_state:
		st.session_state.selected_plot_idx = 0

	st.session_state.selected_session_idx = min(
		max(st.session_state.selected_session_idx, 0),
		max(session_count - 1, 0),
	)


def main() -> None:
	st.set_page_config(page_title="MEG QC Report Navigator", layout="wide")

	css_path = Path(__file__).with_name("style.css")
	with css_path.open(encoding="utf-8") as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

	with st.sidebar:
		st.header("Export QC Report")
		export_slot = st.empty()
		st.header("Dataset")
		root_dir = st.text_input("Output root directory", value=_default_root())
		st.caption("App recursively scans for sub-*/ses-* folders containing qc images.")

		if st.button("Refresh data", width="content"):
			discover_sessions.clear()

		st.header("QC order")
		st.caption("Edit QC_PLOT_ORDER in this script to define order and titles.")
		for idx, item in enumerate(QC_PLOT_ORDER, start=1):
			st.write(f"{idx}. {item['title']} ({item['file_hint']})")

	sessions = discover_sessions(root_dir)
	total_subjects = len({session.subject for session in sessions})
	total_sessions = len(sessions)
	report_html = build_report_html(sessions, QC_PLOT_ORDER)
	export_slot.download_button(
		label="Export full report",
		data=report_html.encode("utf-8"),
		file_name="meg_qc_report.html",
		mime="text/html",
		use_container_width=True,
		help="Long-format report for all discovered subjects/sessions and QC plots.",
	)

	_ensure_state(len(sessions))

	if not sessions:
		st.warning("No sessions with QC plots found. Check your output path and folder structure.")
		return

	left_col, right_col = st.columns([1.15, 2.85], gap="large")

	with left_col:
		st.markdown(
			f"""
			<div class="report-header">
				<h1>MEG Quality Check Navigator</h1>
				<p>Navigate subject/session and review QC plots in your predefined order.</p>
				<div class="report-header-total">Total subjects: {total_subjects}<br>Total sessions: {total_sessions}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		selected_idx = st.selectbox(
			"Subject / session",
			options=list(range(len(sessions))),
			index=st.session_state.selected_session_idx,
			format_func=lambda i: sessions[i].key,
		)

		st.session_state.selected_session_idx = selected_idx
		selected = sessions[selected_idx]

		ordered_plots = sort_files_by_qc_order(selected.qc_files, QC_PLOT_ORDER)
		if not ordered_plots:
			st.info("No QC images found for selected session.")
			return

		st.session_state.selected_plot_idx = min(
			st.session_state.selected_plot_idx,
			len(ordered_plots) - 1,
		)

		# Push navigation controls lower in the left panel for a bottom-aligned layout feel.
		st.markdown("<div class='left-nav-spacer'></div>", unsafe_allow_html=True)

		nav_left, nav_right = st.columns([1, 1])
		with nav_left:
			if st.button("Previous", use_container_width=True):
				st.session_state.selected_plot_idx = max(st.session_state.selected_plot_idx - 1, 0)
				st.rerun()
		with nav_right:
			if st.button("Next", use_container_width=True):
				st.session_state.selected_plot_idx = min(
					st.session_state.selected_plot_idx + 1,
					len(ordered_plots) - 1,
				)
				st.rerun()

	current_title, current_plot = ordered_plots[st.session_state.selected_plot_idx]

	with right_col:

		jump_idx = st.selectbox(
			"All plots",
			options=list(range(len(ordered_plots))),
			index=st.session_state.selected_plot_idx,
			format_func=lambda i: f"{i + 1} - {ordered_plots[i][0]}",
			label_visibility="collapsed",
			width="stretch",
		)
		if jump_idx != st.session_state.selected_plot_idx:
			st.session_state.selected_plot_idx = jump_idx
			st.rerun()

		st.markdown(f"### {current_title}")
		st.image(str(current_plot), use_container_width=True)
		st.caption(str(current_plot))


if __name__ == "__main__":
	main()
