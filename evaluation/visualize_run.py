import json
import os
import webbrowser
from pathlib import Path

import pydot

from evaluation.utils import new_file


import json
import os
import webbrowser
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import pydot

from evaluation.utils import new_file


class HTMLReport:
    BEST_CLASS = "best"
    MODEL_CLASS = "model"
    ROUND_CLASS = "round"

    def __init__(self, title="Report", output_dir="reports", filename="report.html"):
        self.title = title
        self.output_dir = Path(new_file(Path(output_dir) / title))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "img", exist_ok=False)
        self.filename = filename
        self.sections = []  # To store content for different sections
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader("evaluation/report_templates"),
            autoescape=True
        )
        self.html_template = self.env.get_template("report.html")
        self.js_template = self.env.get_template("script.js")
        self.css_template = self.env.get_template("style.css")

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

    def _generate_hypothesis_html(self, num_states: str, round_name: str, hyp_img: str, hyp_stoc_img: str, score: float, is_best: bool,
                                  always_show_info: dict, detailed_info: dict):
        cls = f"{self.MODEL_CLASS} {self.BEST_CLASS if is_best else ''}"
        html = f'<div class="{cls}">'
        html += f'<h3>{num_states} states</h3>'
        html += f'<img src="{hyp_img}" alt="{num_states}-state hypothesis" class="hyp" style="display: block;">'
        html += f'<img src="{hyp_stoc_img}" alt="{num_states}-state hypothesis (stochastic)" class="hyp_stoc" style="display: none;">'
        for key, val in always_show_info.items():
            html += f'<p>{key}: {val}</p>'
        if detailed_info:
            html += self._generate_expandable_info("Additional Details", round_name=round_name, num_states=num_states, info=detailed_info)
        html += "</div>"
        return html

    def add_learning_round(self, round_name: str, data: dict):
        """Adds a visualization of a single learning round to the report."""
        hyps = data["hyp"]
        hyps_stoc = data["hyp_stoc"]
        scores = data.get("heuristic_scores", None)

        round_html = f'<div class="{self.ROUND_CLASS}"><h2>Round {round_name}</h2>'

        if "traces_used_to_learn" in data:
            round_html += self._add_expandible_traces("traces_used_to_learn", round_name, data["traces_used_to_learn"])

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

    def save(self, open_automatically=True):
        html_content = self._generate_full_html()
        file_path = self.output_dir / self.filename
        with open(file_path, "w") as f:
            f.write(html_content)

        if open_automatically:
            webbrowser.open(f"file://{file_path.as_posix()}")


def main():
    # data_path = Path(r"C:\private\Uni\MastersThesis\active-pmsat-inference-wip\learning_results_143\2025-01-27_03-12-35_learning_results_MooreMachine_5States_3Inputs_3Outputs_03b9d56c7ad14407ad5b30abf2263612.dot_APMSL()_5ddf5579.json")
    data_path = Path(r"C:\private\Uni\MastersThesis\active-pmsat-inference-wip\2025-01-27_16-52-57_learning_results_coffeemachine_moore.dot.json")
    if not data_path.exists():
        raise FileNotFoundError(data_path.as_posix())

    with open(data_path, "r") as file:
        data = json.load(file)

    # Create the report
    report = HTMLReport(title="Learning Rounds Report")

    # report.visualize_original_automaton(data["original_automaton"])

    # Add rounds to the report
    for round_name, round_data in data["detailed_learning_info"].items():
        report.add_learning_round(round_name, round_data)

    # Save the report
    report.save()


if __name__ == "__main__":
    main()
