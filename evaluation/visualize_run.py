import json
import os
import webbrowser
from pathlib import Path

import aalpy.utils
import pydot
from jinja2 import Environment, FileSystemLoader
import argparse

from evaluation.utils import new_file
from f_similarity import f_score


class HTMLReport:
    BEST_CLASS = "best"
    MODEL_CLASS = "model"
    ROUND_CLASS = "round"

    def __init__(self, data_path: str, title="Report", output_dir="reports", filename="report.html", log_file=None,
                 original_automaton_file=None):
        """Initialize the report, automatically loads data from the provided data_path."""
        self.title = title
        self.data_path = Path(data_path)
        self.output_dir = Path(new_file(Path(output_dir) / title))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "img", exist_ok=False)
        self.filename = self.output_dir / filename
        self.sections = []  # To store content for different sections

        if log_file is not None:
            with open(log_file, "r") as f:
                self.logs = f.readlines()
        else:
            self.logs = []

        if original_automaton_file is not None:
            self.original_automaton = aalpy.utils.load_automaton_from_file(original_automaton_file, "moore")
        else:
            self.original_automaton = None

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader("evaluation/report_templates"),
            autoescape=True
        )
        self.html_template = self.env.get_template("report.html")
        self.js_template = self.env.get_template("script.js")
        self.css_template = self.env.get_template("style.css")

        self._load_data()
        # Add rounds to the report
        for round_name, round_data in self.data["detailed_learning_info"].items():
            self._generate_learning_round_html(round_name, round_data)
        self.save()

    def _load_data(self):
        """Loads the data from the provided file path."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path.as_posix()} does not exist.")

        with open(self.data_path, "r") as file:
            self.data = json.load(file)

    def _generate_full_html(self):
        """Returns the full HTML with all sections."""
        return self.html_template.render(
            title=self.title,
            sections=''.join(self.sections),
            js_script=self.js_template.render(),
            css_styles=self.css_template.render()
        )

    def _generate_hypothesis_image(self, round_name, num_states, hyp_data, stoc=False):
        """Helper function to generate hypothesis images."""
        hyp_graph, = pydot.graph_from_dot_data(hyp_data)
        hyp_img = f"img/{round_name}_hyp{'_stoc' if stoc else ''}{num_states}.png"
        hyp_graph.write(path=self.output_dir / hyp_img, format="png")
        return hyp_img

    def _generate_expandable_info(self, section_name: str, round_name: str, num_states: str, info: dict):
        """Generates the HTML for an expandable information section containing multiple key-value pairs."""
        expandable_id = f"{section_name.replace(' ', '_')}_expandable_{round_name}_{num_states}"

        # Create a content string for all key-value pairs
        expandable_html = f"""
        <div class="expandable">
            <div class="expandable-header" onclick="toggleExpandable('{expandable_id}')">
                <span>{section_name}</span>
            </div>
            <div class="expandable-content" id="{expandable_id}" style="display: none;">
        """

        # Add each key-value pair to the expandable content
        for key, val in info.items():
            expandable_html += f'<p>{key}: {val}</p>'

        expandable_html += "</div></div>"
        return expandable_html

    def _generate_hypothesis_html(self, num_states: str, round_name: str, hyp_img: str, hyp_stoc_img: str, score: float,
                                  is_best: bool,
                                  always_show_info: dict, detailed_info: dict):
        cls = f"{self.MODEL_CLASS} {self.BEST_CLASS if is_best else ''}"
        html = f'<div class="{cls}">'
        html += f'<h3>{num_states} states</h3>'
        html += f'<img src="{hyp_img}" alt="{num_states}-state hypothesis" class="hyp" style="display: none;">'
        html += f'<img src="{hyp_stoc_img}" alt="{num_states}-state hypothesis (stochastic)" class="hyp_stoc" style="display: block;">'
        for key, val in always_show_info.items():
            html += f'<p>{key}: {val}</p>'
        if detailed_info:
            html += self._generate_expandable_info("Additional Details", round_name=round_name, num_states=num_states,
                                                   info=detailed_info)
        html += "</div>"
        return html

    def _get_logs_of_round(self, round_name: str | None):
        if not self.logs:
            return []

        round = int(round_name or 0)

        start_index = None
        end_index = None
        for i, msg in enumerate(self.logs):
            if f"Starting learning round {round}" in msg:
                start_index = i
            if f"Starting learning round {round + 1}" in msg:
                assert start_index is not None
                end_index = i
                break

        return self.logs[start_index:end_index]


    def _generate_expandible_log(self, round_name: str):
        expandable_id = f"log_expandable_{round_name}"

        # Create a content string for all key-value pairs
        expandable_html = f"""
                <div class="expandable">
                    <div class="expandable-header" onclick="toggleExpandable('{expandable_id}')">
                        <span>Log</span>
                    </div>
                    <div class="expandable-content" id="{expandable_id}" style="display: none;">
                """

        # Add each key-value pair to the expandable content
        for msg in self._get_logs_of_round(round_name):
            expandable_html += f'<p>{msg}</p>'

        expandable_html += "</div></div>"
        return expandable_html

    def _generate_learning_round_html(self, round_name: str, data: dict):
        """Adds a visualization of a single learning round to the report."""
        hyps = data["hyp"]
        hyps_stoc = data["hyp_stoc"]
        scores = data.get("heuristic_scores", None)

        round_html = f'<div class="{self.ROUND_CLASS}"><h2>Round {round_name}</h2>'

        if "traces_used_to_learn" in data:
            round_html += self._add_expandible_traces("traces_used_to_learn", round_name, data["traces_used_to_learn"])

        if self.logs:
            round_html += self._generate_expandible_log(round_name)

        round_html += '<div class="models">'
        for num_states, hyp in hyps.items():
            hyp_stoc = hyps_stoc[num_states]
            is_best = False
            score = None
            if scores is not None:
                score = scores[num_states]
                if num_states == max(scores, key=scores.get):
                    is_best = True

            hyp_img = self._generate_hypothesis_image(round_name, num_states, hyp)
            hyp_stoc_img = self._generate_hypothesis_image(round_name, num_states,
                                                           hyp_stoc.replace('label="!', 'color=red, label="'),
                                                           stoc=True)

            hyp_info = dict()
            if score is not None:
                hyp_info["Score"] = score
            hyp_info["Glitch Percentage"] = f'{data["pmsat_info"][num_states]["percent_glitches"]:.2f}%'
            if self.original_automaton is not None:
                hyp_info["[META] F-Score with original"] = f"{f_score(self.original_automaton, aalpy.utils.FileHandler.load_automaton_from_string(hyp, 'moore')):.2f}"  # not (yet?) merged into aalpy

            hyp_detail_info = dict()
            hyp_detail_info["glitch_trans"] = data["pmsat_info"][num_states]["glitch_trans"]
            hyp_detail_info["dominant_reachable_states"] = data["pmsat_info"][num_states]["dominant_reachable_states"]
            hyp_detail_info["glitched_delta_freq"] = data["pmsat_info"][num_states]["glitched_delta_freq"]
            hyp_detail_info["dominant_delta_freq"] = data["pmsat_info"][num_states]["dominant_delta_freq"]
            hyp_detail_info["solve_time"] = data["pmsat_info"][num_states]["solve_time"]

            round_html += self._generate_hypothesis_html(num_states, round_name, hyp_img, hyp_stoc_img, score, is_best,
                                                         hyp_info, hyp_detail_info)

        round_html += "</div>"

        for additional_traces_name in [k for k in data.keys() if k.startswith("additional_traces")]:
            traces = data.get(additional_traces_name, [])
            if traces:
                round_html += self._add_expandible_traces(additional_traces_name, round_name, traces)
        round_html += "</div>"

        self.sections.append(round_html)

    def _add_expandible_traces(self, traces_name, round_name, traces):
        unique_id = f"{traces_name}_{round_name}"  # Generate a unique ID for each section

        html = ""
        html += f'<div class="expandable">'
        html += f'<div class="expandable-header" onclick="toggleExpandable(\'{unique_id}\')">'
        html += f'<span>{traces_name}</span></div>'
        html += f'<div class="expandable-content" id="{unique_id}" style="display: none;">'
        for trace in traces:
            html += f'<p>{trace}</p>'
        html += '</div>'
        html += f"</div>"
        return html

    def save(self):
        html_content = self._generate_full_html()
        with open(self.filename, "w") as f:
            f.write(html_content)

    def open(self):
        webbrowser.open(f"file://{self.filename.as_posix()}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate an HTML report from learning data.")
    parser.add_argument("data_path", type=str, help="Path to the learning data JSON file.")
    parser.add_argument("--title", type=str, default="Learning Report", help="Title of the report.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save the report.")
    parser.add_argument("--filename", type=str, default="report.html", help="Filename for the report.")
    parser.add_argument("--log-file", type=str, default=None, help="Log file")
    parser.add_argument("--open", type=bool, default=True, help="Open automatically")

    args = parser.parse_args()

    report = HTMLReport(data_path=args.data_path,
                        title=args.title,
                        output_dir=args.output_dir,
                        filename=args.filename,
                        log_file=args.log_file)
    if args.open:
        report.open()


if __name__ == "__main__":
    main()
